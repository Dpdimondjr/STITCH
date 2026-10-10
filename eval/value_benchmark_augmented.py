"""Augmented feature benchmark for STITCH.

Extends eval/value_benchmark.py to test physically-motivated feature blocks:
  1. Baseline: same 16 continuous features as the clean benchmark
  2. +Thermal: adds focal_plane_temp, focal_plane_temp_std (98% fill)
  3. +Aperture: adds flfrcsap, aperture_loss, temp×aperture_loss interaction
  4. Complete-case +Background: adds bkg_frac, bkg_rms_frac, scatter_flag_frac
     (only for the ~31% of rows that have background measurements)

All comparisons are matched: each model is evaluated on the same test stars and
same rows as its baseline. Features are imputed with training-set medians.

Run from the repository root:
    python3 eval/value_benchmark_augmented.py
    python3 eval/value_benchmark_augmented.py --data training_data_v3.parquet
    python3 eval/value_benchmark_augmented.py --out eval/value_benchmark_augmented_results
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# Same 16-column baseline as value_benchmark.py
BASELINE_CONT = [
    "col", "row", "delta_sub_col", "delta_sub_row", "sector", "tmag",
    "crowdsap", "cdpp1_0", "pdcvar", "jitter_rms", "pdc_noi", "pr_wght2",
    "gaiarp", "perstar_gaia_offset", "sector_ccd_mean_loo", "spatial_knn_mean_loo",
]

THERMAL_EXTRA = ["focal_plane_temp", "focal_plane_temp_std"]
APERTURE_EXTRA = ["flfrcsap", "apt_loss", "apt_loss_x_temp"]
BACKGROUND_EXTRA = ["bkg_frac", "bkg_rms_frac", "scatter_flag_frac"]
# PRF capture-fraction features (100% fill when prf parquet is used)
PRF_EXTRA = [
    "prf_q", "prf_dq_dcol", "prf_dq_drow",
    "prf_d2q_dcol2", "prf_d2q_drow2", "prf_boundary_frac",
]
PRF_LOG_RATIO = ["prf_q_log_ratio"]  # 31.6% fill — imputed with median


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="training_data_v3.parquet")
    p.add_argument("--out", default="eval/value_benchmark_augmented_results")
    p.add_argument("--seed", type=int, default=20260915)
    p.add_argument("--train-stars", type=int, default=6000)
    p.add_argument("--val-stars", type=int, default=1500)
    p.add_argument("--test-stars", type=int, default=3000)
    p.add_argument("--bootstrap", type=int, default=1000)
    return p.parse_args()


def cv(frame, offsets):
    """Coefficient of variation of corrected sector_median per star."""
    work = frame[["tic_id", "sector_median"]].copy()
    work["corrected"] = work.sector_median.to_numpy() / offsets
    g = work.groupby("tic_id").corrected
    return (g.std(ddof=0) / g.mean()).sort_index()


def attach_references(ref, query, k=20):
    """Attach KNN mean and CCD mean LOO features from the reference catalog."""
    q = query.copy().reset_index(drop=True)
    q["sector_ccd_mean_loo"] = np.nan
    q["spatial_knn_mean_loo"] = np.nan
    q["reference_count"] = 0
    groups = {key: r for key, r in ref.groupby(["sector", "cam", "ccd"])}
    for key, group in q.groupby(["sector", "cam", "ccd"]):
        r = groups.get(key)
        if r is None or len(r) < 5:
            continue
        _, idx = cKDTree(r[["col", "row"]].to_numpy()).query(
            group[["col", "row"]].to_numpy(), k=min(k, len(r)))
        idx = np.asarray(idx).reshape(len(group), -1)
        values = r["loo"].to_numpy()[idx]
        q.loc[group.index, "spatial_knn_mean_loo"] = values.mean(axis=1)
        q.loc[group.index, "sector_ccd_mean_loo"] = r["loo"].mean()
        q.loc[group.index, "reference_count"] = len(r)
    return q


def build_features(frame, cont_cols):
    """Assemble feature matrix: continuous columns + cam/ccd one-hot."""
    X_cont = frame[cont_cols].to_numpy(dtype=float)
    X_cam = np.eye(4)[frame.cam.astype(int).to_numpy() - 1]
    X_ccd = np.eye(4)[frame.ccd.astype(int).to_numpy() - 1]
    return np.column_stack([X_cont, X_cam, X_ccd])


def add_derived_columns(df):
    """Add derived physical features from existing columns in-place."""
    df["apt_loss"] = 1.0 - df["flfrcsap"]
    df["apt_loss_x_temp"] = df["focal_plane_temp"] * df["apt_loss"]
    df["bkg_frac"] = np.where(
        df["sector_median"] > 0,
        df["median_sap_bkg"] / df["sector_median"],
        np.nan,
    )
    df["bkg_rms_frac"] = np.where(
        df["sector_median"] > 0,
        df["bkg_rms"] / df["sector_median"],
        np.nan,
    )
    return df


def train_hgb(xtr, ytr, xv, yv, seed):
    """Validation-tuned HGB. Returns (model, val_score)."""
    best_model, best_score = None, np.inf
    for leaves in [15, 31, 63]:
        m = HistGradientBoostingRegressor(
            max_leaf_nodes=leaves, max_iter=150, l2_regularization=1.0,
            random_state=seed, early_stopping=False)
        m.fit(xtr, ytr)
        pv = np.clip(m.predict(xv), 0.5, 2.0)
        score = float((cv(None, pv) if False else
                        pd.Series(yv).sub(pd.Series(pv)).abs().median()))
        # Use validation MAE as proxy score for tuning
        score = float(np.median(np.abs(yv - np.clip(m.predict(xv), 0.5, 2.0))))
        if score < best_score:
            best_score, best_model = score, m
    return best_model


def eval_model(model, prep, te, raw_cv, label, rng, n_bootstrap):
    """Evaluate a model: returns reduction % and bootstrap CI."""
    xt = prep.transform(build_features(te, prep._cont_cols))
    preds = np.clip(model.predict(xt), 0.5, 2.0)
    corrected_cv = cv(te, preds)
    raw_arr = raw_cv.loc[corrected_cv.index].to_numpy()
    cor_arr = corrected_cv.to_numpy()
    reduction = 100.0 * (1 - np.median(cor_arr) / np.median(raw_arr))
    boot_idx = rng.integers(0, len(raw_arr), size=(n_bootstrap, len(raw_arr)))
    boot_r = 100.0 * (1 - np.median(cor_arr[boot_idx], axis=1) /
                       np.median(raw_arr[boot_idx], axis=1))
    ci = np.quantile(boot_r, [0.025, 0.975])
    worse_frac = 100.0 * np.mean(cor_arr > raw_arr)
    return dict(label=label, median_cv=float(100 * np.median(cor_arr)),
                reduction=float(reduction),
                ci_lo=float(ci[0]), ci_hi=float(ci[1]),
                stars_worse=float(worse_frac),
                n_stars=len(cor_arr))


class Preprocessor:
    """Fits imputer + scaler on training data; stores cont_cols for rebuild."""
    def __init__(self, cont_cols):
        self._cont_cols = cont_cols
        self._pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())

    def fit_transform(self, frame):
        X = build_features(frame, self._cont_cols)
        return self._pipe.fit_transform(X)

    def transform(self, frame):
        X = build_features(frame, self._cont_cols)
        return self._pipe.transform(X)


def main():
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    print(f"Loading {args.data} …")
    df = pd.read_parquet(args.data)

    # ── Selection (same as value_benchmark.py) ────────────────────────────────
    valid = (df.sector.between(1, 100) & df.cam.isin([1, 2, 3, 4]) &
             df.ccd.isin([1, 2, 3, 4]) & df.tmag.le(13) &
             df.sector_median.gt(0) & df.col.between(0, 2200) &
             df.row.between(0, 2200) & df.gaiarp.notna())
    df = df[valid].copy()
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["sector_median", "gaiarp"])
    counts = df.groupby("tic_id").sector.nunique()
    df = df[df.tic_id.isin(counts[counts >= 5].index)].copy()
    print(f"  {len(df):,} rows; {df.tic_id.nunique():,} stars after selection")

    # ── Add derived columns ───────────────────────────────────────────────────
    df = add_derived_columns(df)

    # ── Split (same seed/method as value_benchmark.py) ───────────────────────
    stars = df.groupby("tic_id").cam.agg(lambda x: int(x.mode().iloc[0]))
    refs, rest = train_test_split(stars.index.to_numpy(), test_size=0.6,
                                  stratify=stars.to_numpy(), random_state=args.seed)
    train, tmp = train_test_split(rest, test_size=1 / 3, stratify=stars.loc[rest],
                                  random_state=args.seed)
    val, test = train_test_split(tmp, test_size=0.5, stratify=stars.loc[tmp],
                                 random_state=args.seed)

    def cap(ids, n):
        return rng.choice(ids, min(n, len(ids)), replace=False)

    train_ids = cap(train, args.train_stars)
    val_ids = cap(val, args.val_stars)
    test_ids = cap(test, args.test_stars)

    # ── Recompute LOO from raw sector_median ──────────────────────────────────
    g = df.groupby("tic_id").sector_median
    total, n = g.transform("sum"), g.transform("count")
    df["loo"] = df.sector_median / ((total - df.sector_median) / (n - 1))

    # ── Per-star Gaia anchor (from reference stars only) ─────────────────────
    ref_df = df[df.tic_id.isin(refs)].copy()
    cal = ref_df.groupby(["tic_id", "cam"]).agg(
        flux=("sector_median", "median"), rp=("gaiarp", "first")).reset_index()
    coefs = {}
    df["gaia"] = np.nan
    for cam, r in cal.groupby("cam"):
        slope, intercept = np.polyfit(r.rp, np.log10(r.flux), 1)
        coefs[int(cam)] = [float(slope), float(intercept)]
        mask = df.cam == cam
        df.loc[mask, "gaia"] = (df.loc[mask, "sector_median"] /
                                 10 ** (slope * df.loc[mask, "gaiarp"] + intercept))
    df["perstar_gaia_offset"] = df.groupby("tic_id").gaia.transform("mean")

    # ── Attach KNN reference features (from reference catalog only) ───────────
    print("  Attaching KNN reference features …")
    ref_df = df[df.tic_id.isin(refs)].copy()

    def make_partition(ids):
        q = attach_references(ref_df, df[df.tic_id.isin(ids)])
        ns = q.groupby("tic_id").sector.nunique()
        q = q[q.tic_id.isin(ns[ns >= 5].index)].dropna(
            subset=["spatial_knn_mean_loo"]).reset_index(drop=True)
        return q

    tr = make_partition(train_ids)
    vl = make_partition(val_ids)
    te = make_partition(test_ids)
    print(f"  Train: {tr.tic_id.nunique():,} stars | Val: {vl.tic_id.nunique():,} | "
          f"Test: {te.tic_id.nunique():,}")

    raw_cv = cv(te, np.ones(len(te)))

    results = []

    # ── Experiment A: full population ─────────────────────────────────────────
    for block_name, cont_cols in [
        ("Baseline (no thermal)", BASELINE_CONT),
        ("+Thermal (temp+temp_std)", BASELINE_CONT + THERMAL_EXTRA),
        ("+Aperture (flfrcsap+apt_loss+interaction)", BASELINE_CONT + THERMAL_EXTRA + APERTURE_EXTRA),
    ]:
        print(f"\n  Training HGB [{block_name}] …")
        prep = Preprocessor(cont_cols)
        xtr = prep.fit_transform(tr)
        xv = prep.transform(vl)
        xt = prep.transform(te)
        ytr = tr["loo"].to_numpy()
        yv = vl["loo"].to_numpy()
        yt = te["loo"].to_numpy()

        model = train_hgb(xtr, ytr, xv, yv, args.seed)
        preds_t = np.clip(model.predict(xt), 0.5, 2.0)
        corr_cv = cv(te, preds_t)
        raw_a = raw_cv.loc[corr_cv.index].to_numpy()
        cor_a = corr_cv.to_numpy()
        reduction = 100.0 * (1 - np.median(cor_a) / np.median(raw_a))
        boot_idx = rng.integers(0, len(raw_a), size=(args.bootstrap, len(raw_a)))
        boot_r = 100.0 * (1 - np.median(cor_a[boot_idx], axis=1) /
                           np.median(raw_a[boot_idx], axis=1))
        ci = np.quantile(boot_r, [0.025, 0.975])
        results.append(dict(
            regime="Full population",
            label=block_name,
            n_stars=len(cor_a),
            median_cv_pct=float(100 * np.median(cor_a)),
            reduction_pct=float(reduction),
            ci_lo=float(ci[0]), ci_hi=float(ci[1]),
            stars_worse_pct=float(100 * np.mean(cor_a > raw_a)),
        ))
        print(f"    {block_name}: {reduction:.1f}% reduction "
              f"(95% CI {ci[0]:.1f}, {ci[1]:.1f})  "
              f"median CV {100*np.median(cor_a):.3f}%")

    # ── Experiment B: complete-case (background subset) ───────────────────────
    bkg_cols = ["flfrcsap", "median_sap_bkg", "bkg_rms", "scatter_flag_frac"]
    has_bkg = te[bkg_cols].notna().all(axis=1)
    te_cc = te[has_bkg].copy()
    stars_cc = te_cc.tic_id.unique()
    tr_cc = tr[tr.tic_id.isin(stars_cc) | tr.tic_id.notna()].copy()  # keep all training
    vl_cc = vl[vl.tic_id.isin(stars_cc) | vl.tic_id.notna()].copy()
    # For matched comparison, only evaluate on stars that have background data in test
    # Use all training data but evaluate on complete-case test subset
    raw_cv_cc = cv(te_cc, np.ones(len(te_cc)))

    print(f"\n  Complete-case test subset: {te_cc.tic_id.nunique():,} test stars "
          f"with background data ({len(te_cc):,} rows)")

    bkg_cont_list = BASELINE_CONT + THERMAL_EXTRA + APERTURE_EXTRA + BACKGROUND_EXTRA
    for block_name, cont_cols, te_eval in [
        ("CC Baseline", BASELINE_CONT, te_cc),
        ("CC +All physical features", bkg_cont_list, te_cc),
    ]:
        print(f"\n  Training HGB [{block_name}] on complete-case subset …")
        # Train on subset of training rows that also have background data for fair comparison
        tr_has_bkg = tr[tr[bkg_cols].notna().all(axis=1)].copy()
        vl_has_bkg = vl[vl[bkg_cols].notna().all(axis=1)].copy()

        if len(tr_has_bkg) < 1000 or tr_has_bkg.tic_id.nunique() < 100:
            print(f"    Skipping {block_name}: insufficient training rows with background data")
            continue

        prep = Preprocessor(cont_cols)
        xtr = prep.fit_transform(tr_has_bkg)
        xv = prep.transform(vl_has_bkg) if len(vl_has_bkg) > 10 else prep.transform(vl[:10])
        xt = prep.transform(te_eval)
        ytr = tr_has_bkg["loo"].to_numpy()
        yv = vl_has_bkg["loo"].to_numpy() if len(vl_has_bkg) > 10 else vl["loo"].to_numpy()[:10]

        model = train_hgb(xtr, ytr, xv, yv, args.seed)
        preds_t = np.clip(model.predict(xt), 0.5, 2.0)
        corr_cv = cv(te_eval, preds_t)
        raw_a = raw_cv_cc.loc[corr_cv.index].to_numpy()
        cor_a = corr_cv.to_numpy()
        if len(cor_a) == 0:
            print(f"    Skipping {block_name}: no test rows")
            continue
        reduction = 100.0 * (1 - np.median(cor_a) / np.median(raw_a))
        boot_idx = rng.integers(0, len(raw_a), size=(args.bootstrap, len(raw_a)))
        boot_r = 100.0 * (1 - np.median(cor_a[boot_idx], axis=1) /
                           np.median(raw_a[boot_idx], axis=1))
        ci = np.quantile(boot_r, [0.025, 0.975])
        results.append(dict(
            regime="Complete-case (background subset)",
            label=block_name,
            n_stars=len(cor_a),
            median_cv_pct=float(100 * np.median(cor_a)),
            reduction_pct=float(reduction),
            ci_lo=float(ci[0]), ci_hi=float(ci[1]),
            stars_worse_pct=float(100 * np.mean(cor_a > raw_a)),
        ))
        print(f"    {block_name}: {reduction:.1f}% reduction "
              f"(95% CI {ci[0]:.1f}, {ci[1]:.1f})  "
              f"median CV {100*np.median(cor_a):.3f}%")

    # ── Experiment C: PRF features (only when columns are present) ───────────
    prf_cols_present = all(c in te.columns for c in PRF_EXTRA)
    if prf_cols_present:
        print("\n── Experiment C: PRF capture-fraction features ──")
        for block_name, cont_cols in [
            ("Baseline (no PRF)", BASELINE_CONT),
            ("+PRF capture fraction", BASELINE_CONT + PRF_EXTRA),
            ("+PRF + log-ratio", BASELINE_CONT + PRF_EXTRA + PRF_LOG_RATIO),
        ]:
            print(f"\n  Training HGB [{block_name}] …")
            prep = Preprocessor(cont_cols)
            xtr = prep.fit_transform(tr)
            xv = prep.transform(vl)
            xt = prep.transform(te)
            ytr = tr["loo"].to_numpy()
            yv = vl["loo"].to_numpy()

            model = train_hgb(xtr, ytr, xv, yv, args.seed)
            preds_t = np.clip(model.predict(xt), 0.5, 2.0)
            corr_cv = cv(te, preds_t)
            raw_a = raw_cv.loc[corr_cv.index].to_numpy()
            cor_a = corr_cv.to_numpy()
            reduction = 100.0 * (1 - np.median(cor_a) / np.median(raw_a))
            boot_idx = rng.integers(0, len(raw_a), size=(args.bootstrap, len(raw_a)))
            boot_r = 100.0 * (1 - np.median(cor_a[boot_idx], axis=1) /
                               np.median(raw_a[boot_idx], axis=1))
            ci = np.quantile(boot_r, [0.025, 0.975])
            results.append(dict(
                regime="PRF features",
                label=block_name,
                n_stars=len(cor_a),
                median_cv_pct=float(100 * np.median(cor_a)),
                reduction_pct=float(reduction),
                ci_lo=float(ci[0]), ci_hi=float(ci[1]),
                stars_worse_pct=float(100 * np.mean(cor_a > raw_a)),
            ))
            print(f"    {block_name}: {reduction:.1f}% reduction "
                  f"(95% CI {ci[0]:.1f}, {ci[1]:.1f})  "
                  f"median CV {100*np.median(cor_a):.3f}%")
    else:
        print("\n  Skipping PRF experiment: prf_* columns not in parquet. "
              "Run: python3 data/compute_prf_features.py training_data_v3.parquet "
              "training_data_v3_prf.parquet")

    # ── Save results ──────────────────────────────────────────────────────────
    (out / "results.json").write_text(json.dumps(results, indent=2))

    lines = ["# STITCH augmented feature benchmark", "",
             "HGB trained on matched populations; comparisons are within-regime.",
             "",
             "| Regime | Feature block | N stars | Median CV % | Reduction % (95% CI) | Stars worse % |",
             "|---|---|---:|---:|---:|---:|"]
    for r in results:
        lines.append(
            f"| {r['regime']} | {r['label']} | {r['n_stars']:,} | "
            f"{r['median_cv_pct']:.3f} | {r['reduction_pct']:.1f} "
            f"({r['ci_lo']:.1f}, {r['ci_hi']:.1f}) | {r['stars_worse_pct']:.1f} |"
        )
    lines += ["",
              "Notes:",
              "- Baseline matches eval/value_benchmark.py CONT features.",
              "- All HGB models are validation-tuned on {15, 31, 63} max_leaf_nodes.",
              "- Bootstrap over test stars; model and reference catalog are fixed.",
              "- Complete-case regime uses only observations with background data for"
              " both training and evaluation — do not compare medians to full-population rows.",
              "- PRF regime: prf_* features from data/compute_prf_features.py (training_data_v3_prf.parquet).",
              "  Requires: python3 eval/value_benchmark_augmented.py --data training_data_v3_prf.parquet"
              " --out eval/value_benchmark_prf_results"]
    report_text = "\n".join(lines) + "\n"
    (out / "REPORT.md").write_text(report_text)
    print("\n" + report_text)

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, max(4, len(results) * 0.6 + 2)))
    colors = {"Full population": "#4C72B0", "Complete-case (background subset)": "#DD8452"}
    y_pos = list(range(len(results)))
    for i, r in enumerate(results):
        color = colors.get(r["regime"], "#666666")
        val = r["reduction_pct"]
        lo = val - r["ci_lo"]
        hi = r["ci_hi"] - val
        ax.errorbar(val, i, xerr=[[lo], [hi]], fmt="o", color=color, capsize=3, markersize=6)
        ax.text(val + 0.3, i, f"{val:.1f}%", va="center", fontsize=8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([r["label"] for r in results], fontsize=8)
    ax.invert_yaxis()
    ax.axvline(0, color="gray", lw=1)
    ax.set_xlabel("Reduction in median cross-sector scatter (%) with 95% bootstrap CI")
    ax.set_title("Augmented feature benchmark — HGB on STITCH v3\n"
                 "Blue: full population  Orange: complete-case background subset")
    import matplotlib.patches as mpatches
    patches = [mpatches.Patch(color=c, label=l) for l, c in colors.items()]
    ax.legend(handles=patches, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "comparison.png", dpi=150)
    plt.close(fig)
    print(f"Saved → {out}/comparison.png")
    print(f"Saved → {out}/REPORT.md")


if __name__ == "__main__":
    main()
