"""Background-bias diagnostic for STITCH.

Tests whether sector-level background-to-flux ratio and aperture loss predict
LOO flux residuals — distinguishing additive background errors from multiplicative
aperture/throughput errors. Run from the repository root.

Usage:
    python3 eval/bgbias_diagnostic.py
    python3 eval/bgbias_diagnostic.py --data training_data_v3.parquet
    python3 eval/bgbias_diagnostic.py --out eval/my_diagnostic.png
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="training_data_v3.parquet")
    p.add_argument("--out", default="eval/bgbias_diagnostic.png")
    p.add_argument("--min-sectors", type=int, default=5,
                   help="Min sectors with background data for within-star analysis")
    return p.parse_args()


def within_star_corr(group, x_col, y_col):
    """Pearson r between x_col and y_col within a star's sector observations."""
    x = group[x_col].to_numpy()
    y = group[y_col].to_numpy()
    if len(x) < 3:
        return np.nan
    xm, ym = x - x.mean(), y - y.mean()
    denom = np.sqrt((xm**2).sum() * (ym**2).sum())
    if denom < 1e-12:
        return np.nan
    return float(np.dot(xm, ym) / denom)


def main():
    args = parse_args()

    print(f"Loading {args.data} …")
    df = pd.read_parquet(args.data)
    print(f"  {len(df):,} rows, {df.tic_id.nunique():,} unique stars")

    # Restrict to rows with background and aperture data
    has_bkg = df["flfrcsap"].notna() & df["median_sap_bkg"].notna() & df["sector_median"].gt(0)
    sub = df[has_bkg].copy()
    print(f"  {len(sub):,} rows with background data ({len(sub)/len(df)*100:.1f}%)")
    print(f"  {sub.tic_id.nunique():,} unique stars with background data")

    sub["resid"] = sub["flux_offset_loo"] - 1.0
    sub["bkg_frac"] = sub["median_sap_bkg"] / sub["sector_median"]
    sub["apt_loss"] = 1.0 - sub["flfrcsap"]

    # Clip extreme outliers for analysis (keep for real data, but clip for correlation)
    sub_clip = sub[sub["bkg_frac"].between(-0.5, 2.0) & sub["resid"].between(-0.3, 0.3)].copy()
    print(f"  {len(sub_clip):,} rows after clipping bkg_frac and resid extremes")

    # ── 1. Global correlations ────────────────────────────────────────────────
    print("\n── Global correlations with LOO residual ──")
    for col in ["bkg_frac", "apt_loss", "crowdsap"]:
        r = sub_clip["resid"].corr(sub_clip[col])
        print(f"  {col:20s}: r = {r:+.4f}")

    # ── 2. Within-star cross-sector correlation ───────────────────────────────
    print(f"\n── Within-star bkg_frac vs resid correlation (≥{args.min_sectors} sectors) ──")
    star_counts = sub_clip.groupby("tic_id")["bkg_frac"].count()
    eligible = star_counts[star_counts >= args.min_sectors].index
    sub_ws = sub_clip[sub_clip.tic_id.isin(eligible)]
    print(f"  {len(eligible):,} stars with ≥{args.min_sectors} sectors of background data")

    ws_r = (sub_ws.groupby("tic_id")
            .apply(lambda g: within_star_corr(g, "bkg_frac", "resid"), include_groups=False)
            .dropna())
    print(f"  Within-star r: median={ws_r.median():.3f}  mean={ws_r.mean():.3f}  "
          f"std={ws_r.std():.3f}")
    print(f"  |r| > 0.30: {(ws_r.abs() > 0.30).mean()*100:.1f}% of stars")
    print(f"  |r| > 0.50: {(ws_r.abs() > 0.50).mean()*100:.1f}% of stars")
    print(f"  Positive r : {(ws_r > 0).mean()*100:.1f}% of stars")

    ws_r_apt = (sub_ws.groupby("tic_id")
                .apply(lambda g: within_star_corr(g, "apt_loss", "resid"), include_groups=False)
                .dropna())
    print(f"  Aperture-loss within-star r: median={ws_r_apt.median():.3f}")

    # ── 3. Binned analysis ────────────────────────────────────────────────────
    print("\n── Binned median |resid| by bkg_frac percentile ──")
    bins = np.nanpercentile(sub_clip["bkg_frac"], np.linspace(0, 100, 11))
    bin_labels = range(10)
    sub_clip["bkg_bin"] = pd.cut(sub_clip["bkg_frac"], bins=bins, labels=bin_labels,
                                  include_lowest=True)
    binned = sub_clip.groupby("bkg_bin")["resid"].agg(["median", lambda x: x.abs().median()])
    binned.columns = ["median_resid", "median_abs_resid"]
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    print(f"  {'bkg_frac range':20s} {'median |resid|':>15s}")
    for i, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
        mar = binned.loc[i, "median_abs_resid"] if i in binned.index else np.nan
        print(f"  [{lo:.3f}, {hi:.3f})         {mar*100:.4f}%")

    # ── 4. Within-detector regression ────────────────────────────────────────
    print("\n── Within-(cam, ccd) Ridge regression: resid ~ [bkg_frac, apt_loss, focal_temp] ──")
    feat_cols = ["bkg_frac", "apt_loss", "focal_plane_temp"]
    ccd_r2 = {}
    for (cam, ccd), g in sub_clip.groupby(["cam", "ccd"]):
        if len(g) < 100:
            continue
        X = g[feat_cols].to_numpy(dtype=float)
        y = g["resid"].to_numpy()
        finite = np.isfinite(X).all(axis=1) & np.isfinite(y)
        X, y = X[finite], y[finite]
        if len(X) < 100:
            continue
        model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                               Ridge(alpha=10.0))
        model.fit(X, y)
        ss_res = ((y - model.predict(X))**2).sum()
        ss_tot = ((y - y.mean())**2).sum()
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        ccd_r2[(cam, ccd)] = r2
        print(f"  cam{int(cam)} ccd{int(ccd)}: R²={r2:.4f}  (n={len(X):,})")

    # ── 5. Summary interpretation ─────────────────────────────────────────────
    median_ws_r = ws_r.median()
    high_r_frac = (ws_r.abs() > 0.30).mean()
    mean_r2 = np.mean(list(ccd_r2.values())) if ccd_r2 else 0.0

    print("\n── Summary ──")
    if median_ws_r > 0.3 and high_r_frac > 0.4:
        print("  SIGNAL: Within-star background-flux correlation is substantial.")
        print("  Background bias may contribute meaningfully to sector offsets.")
        print("  Recommend: test bgbias * N_aperture / sector_median as a feature.")
    elif median_ws_r > 0.1:
        print("  WEAK SIGNAL: Modest positive within-star correlation.")
        print("  Background bias is a minor effect on this bright quiet-star population.")
        print("  Not the primary driver of remaining scatter; prioritise PRF/aperture features.")
    else:
        print("  NO CLEAR SIGNAL: Background fraction does not predict residuals within stars.")
        print("  Multiplicative aperture/throughput errors are a more likely explanation.")
    print(f"  Average within-CCD linear R²: {mean_r2:.4f}")

    # ── 6. Plot ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle("Background-bias diagnostic — STITCH v3 parquet", fontsize=11)

    # Panel 1: binned |resid| vs bkg_frac
    ax = axes[0]
    mar_vals = []
    for i in range(10):
        if i in binned.index and not np.isnan(binned.loc[i, "median_abs_resid"]):
            mar_vals.append(binned.loc[i, "median_abs_resid"])
        else:
            mar_vals.append(np.nan)
    ax.plot(bin_centers, np.array(mar_vals) * 100, "o-", color="#4C72B0")
    ax.set_xlabel("Background fraction (median_sap_bkg / sector_median)")
    ax.set_ylabel("Median |LOO residual| (%)")
    ax.set_title("Scatter vs background fraction")
    ax.grid(alpha=0.3)

    # Panel 2: within-star r distribution
    ax = axes[1]
    ax.hist(ws_r.clip(-1, 1), bins=40, color="#4C72B0", alpha=0.8, edgecolor="white")
    ax.axvline(ws_r.median(), color="red", lw=1.5, linestyle="--",
               label=f"Median r = {ws_r.median():.3f}")
    ax.axvline(0, color="gray", lw=1, linestyle=":")
    ax.set_xlabel("Within-star r(bkg_frac, LOO residual)")
    ax.set_ylabel("Count (stars)")
    ax.set_title(f"Within-star background correlation\n({len(ws_r):,} stars with ≥{args.min_sectors} sectors)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Panel 3: per-CCD R²
    ax = axes[2]
    if ccd_r2:
        labels = [f"cam{int(k[0])}.{int(k[1])}" for k in ccd_r2]
        vals = list(ccd_r2.values())
        ax.barh(labels, vals, color="#4C72B0", alpha=0.8)
        ax.set_xlabel("Ridge R² (bkg_frac + apt_loss + focal_temp → resid)")
        ax.set_title("Within-CCD linear explainability")
        ax.axvline(0, color="gray", lw=1)
        ax.grid(alpha=0.3, axis="x")
    else:
        ax.text(0.5, 0.5, "No CCD groups with ≥100 rows", ha="center", va="center",
                transform=ax.transAxes)

    plt.tight_layout()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
