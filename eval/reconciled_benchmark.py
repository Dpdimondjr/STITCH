"""Reconciled benchmark: all models on the value_benchmark canonical split.

Loads the split from eval/value_benchmark_results/split.csv, rebuilds the
feature pipeline, and evaluates KNN, residual HGB, and robust global HGB.
The NSF results already live in value_benchmark_results/per_star_cv.csv on
the same split and are merged in at the end.

Run from the repository root:
    python3 eval/reconciled_benchmark.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).parent))
from residual_multiscale_experiment import spatial_features

OUT = Path("eval/reconciled_benchmark_results")
SEED = 20260915
# Same selection as value_benchmark.py
MIN_SECTORS = 5
MAX_TMAG = 13
REF_CLIP = (0.5, 1.5)


# ── metric ────────────────────────────────────────────────────────────────────

def cv_per_star(frame, offsets):
    """Per-star coefficient of variation of corrected flux (no clipping)."""
    work = frame[["tic_id", "sector_median"]].copy()
    work["corrected"] = work.sector_median.to_numpy() / offsets
    g = work.groupby("tic_id").corrected
    cv = g.std(ddof=0) / g.mean()
    return cv[work.groupby("tic_id").size() >= 5]


# ── KNN baseline ──────────────────────────────────────────────────────────────

def knn_median(ref, query, k=20):
    """Sector/CCD-matched KNN median on LOO labels from ref (excluding own TIC)."""
    result = np.ones(len(query))
    query = query.reset_index(drop=True)
    for key, qg in query.groupby(["sector", "cam", "ccd"]):
        r = ref[(ref.sector == key[0]) & (ref.cam == key[1]) & (ref.ccd == key[2])]
        if len(r) < 2:
            continue
        coords = r[["col", "row"]].to_numpy(float)
        qcoords = qg[["col", "row"]].to_numpy(float)
        qids = qg.tic_id.to_numpy()
        rids = r.tic_id.to_numpy()
        take = min(k + int(r.tic_id.value_counts().max()), len(r))
        dists, idxs = cKDTree(coords).query(qcoords, k=take)
        if take == 1:
            dists, idxs = dists[:, None], idxs[:, None]
        loo = r.loo.to_numpy()
        for i, (q_id, nn) in enumerate(zip(qids, idxs)):
            keep = rids[nn] != q_id
            vals = loo[nn[keep][:k]]
            if len(vals) >= 1:
                result[qg.index[i]] = np.median(vals)
    return result


# ── robust global model ───────────────────────────────────────────────────────

def group_median(values, ids):
    return pd.Series(values).groupby(pd.Series(ids)).transform("median").to_numpy()


def fit_robust_global(X, log_flux, log_base, ids, rounds=5):
    response = log_flux - log_base
    intercept = group_median(response, ids)
    model = None
    for i in range(rounds):
        model = HistGradientBoostingRegressor(
            loss="absolute_error", learning_rate=0.06, max_iter=250,
            max_leaf_nodes=31, min_samples_leaf=40, l2_regularization=0.1,
            random_state=SEED + i, early_stopping=False)
        model.fit(X, response - intercept)
        intercept = group_median(response - model.predict(X), ids)
    return model


def tune_robust(frame, log_base, residual):
    best = None
    for beta in np.linspace(-0.25, 1.5, 36):
        pred = np.exp(log_base + beta * residual)
        score = float(cv_per_star(frame, pred).median())
        if best is None or score < best[1]:
            best = (float(beta), score)
    return best


# ── residual HGB ──────────────────────────────────────────────────────────────

def tune_alpha(frame, raw_pred, knn):
    best = None
    for alpha in np.linspace(0, 1.25, 26):
        pred = 1 + alpha * (raw_pred - 1)
        score = float(cv_per_star(frame, pred).median())
        if best is None or score < best[1]:
            best = (float(alpha), score)
    return best


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)

    # Load canonical split
    split_csv = Path("eval/value_benchmark_results/split.csv")
    if not split_csv.exists():
        raise FileNotFoundError("Run value_benchmark.py first to generate split.csv")
    split = pd.read_csv(split_csv)
    part = {name: set(g.tic_id) for name, g in split.groupby("partition")}
    print(f"Split loaded: "
          + ", ".join(f"{k}={len(v)}" for k, v in sorted(part.items())),
          flush=True)

    # Load data with same selection as value_benchmark
    print("Loading parquet …", flush=True)
    df = pd.read_parquet("training_data_v3.parquet")
    valid = (df.sector.between(1, 100) & df.cam.isin([1,2,3,4]) &
             df.ccd.isin([1,2,3,4]) & df.tmag.le(MAX_TMAG) &
             df.sector_median.gt(0) & df.col.between(0, 2200) &
             df.row.between(0, 2200) & df.gaiarp.notna())
    df = df[valid].copy().replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["sector_median", "gaiarp"])
    counts = df.groupby("tic_id").sector.nunique()
    df = df[df.tic_id.isin(counts[counts >= MIN_SECTORS].index)].copy()

    # Recompute LOO labels in-sample (same as value_benchmark)
    g = df.groupby("tic_id").sector_median
    total, n = g.transform("sum"), g.transform("count")
    df["loo"] = df.sector_median / ((total - df.sector_median) / (n - 1))

    # Partition frames
    frames = {}
    for name in ["reference", "train", "validation", "test"]:
        frames[name] = df[df.tic_id.isin(part[name])].copy().reset_index(drop=True)
        print(f"  {name}: {len(frames['reference' if name=='reference' else name]):,} rows, "
              f"{frames['reference' if name=='reference' else name].tic_id.nunique():,} stars",
              flush=True)
    # Flatten ref+train for spatial feature construction
    ref_for_features = pd.concat([frames["reference"], frames["train"]], ignore_index=True)
    # Clip reference LOO labels to prevent extreme values from contaminating KNN
    raw_loo = ref_for_features.loo.to_numpy()
    ref_for_features["loo"] = raw_loo.clip(*REF_CLIP)
    clipped_frac = float(np.mean((raw_loo < REF_CLIP[0]) | (raw_loo > REF_CLIP[1])))
    print(f"  Reference LOO clipped: {100*clipped_frac:.3f}%", flush=True)

    # Build spatial features for each partition
    BASE_COLS = ["col", "row", "delta_sub_col", "delta_sub_row", "sector", "cam", "ccd",
                 "tmag", "crowdsap", "cdpp1_0", "pdcvar", "jitter_rms",
                 "pdc_noi", "pr_wght2", "gaiarp"]
    feats = {}
    X = {}
    for name in ["train", "validation", "test"]:
        print(f"Spatial features for {name} …", flush=True)
        feats[name] = spatial_features(ref_for_features, frames[name])
        X[name] = pd.concat(
            [frames[name][BASE_COLS].reset_index(drop=True),
             feats[name].reset_index(drop=True)], axis=1
        )

    # KNN baseline (median, no model)
    print("KNN baseline …", flush=True)
    knn_val = knn_median(ref_for_features, frames["validation"])
    knn_test = knn_median(ref_for_features, frames["test"])

    # Validation-calibrate KNN (scale toward 1)
    best_alpha_knn = None
    for alpha in np.linspace(0, 1.25, 26):
        pred = 1 + alpha * (knn_val - 1)
        score = float(cv_per_star(frames["validation"], pred).median())
        if best_alpha_knn is None or score < best_alpha_knn[1]:
            best_alpha_knn = (float(alpha), score)
    knn_test_calibrated = 1 + best_alpha_knn[0] * (knn_test - 1)
    print(f"  KNN val-calibrated alpha={best_alpha_knn[0]:.3f}", flush=True)

    # Residual HGB (LOO target)
    print("Residual HGB …", flush=True)
    knn_train = knn_median(ref_for_features, frames["train"])
    train_target = frames["train"].loo.to_numpy() - knn_train
    best_model = None
    for leaves in [15, 31, 63]:
        m = HistGradientBoostingRegressor(
            loss="absolute_error", learning_rate=0.06, max_iter=250,
            max_leaf_nodes=leaves, min_samples_leaf=40, l2_regularization=0.1,
            random_state=SEED, early_stopping=False)
        m.fit(X["train"], train_target)
        raw_pred = knn_val + m.predict(X["validation"])
        score = float(cv_per_star(frames["validation"], raw_pred).median())
        if best_model is None or score < best_model[2]:
            best_model = (m, leaves, score)
    res_hgb, best_leaves, _ = best_model
    raw_hgb_test = knn_test + res_hgb.predict(X["test"])
    alpha_hgb, _ = tune_alpha(frames["validation"],
                               knn_val + res_hgb.predict(X["validation"]), knn_val)
    hgb_test = 1 + alpha_hgb * (raw_hgb_test - 1)
    print(f"  Residual HGB best leaves={best_leaves}, alpha={alpha_hgb:.3f}", flush=True)

    # Robust global model
    print("Robust global model …", flush=True)
    log_flux_train = np.log(frames["train"].sector_median.to_numpy())
    log_base_train = np.log(np.clip(knn_train, 1e-6, None))
    robust_model = fit_robust_global(
        X["train"], log_flux_train, log_base_train,
        frames["train"].tic_id.to_numpy())
    rv = robust_model.predict(X["validation"])
    beta, _ = tune_robust(frames["validation"],
                           np.log(np.clip(knn_val, 1e-6, None)), rv)
    robust_test = np.exp(np.log(np.clip(knn_test, 1e-6, None))
                         + beta * robust_model.predict(X["test"]))
    print(f"  Robust global beta={beta:.3f}", flush=True)

    # Per-star CV on test set
    test_frame = frames["test"]
    raw_cv = cv_per_star(test_frame, np.ones(len(test_frame)))
    knn_cv = cv_per_star(test_frame, knn_test_calibrated)
    hgb_cv = cv_per_star(test_frame, hgb_test)
    rg_cv = cv_per_star(test_frame, robust_test)
    common = raw_cv.index.intersection(knn_cv.index).intersection(
        hgb_cv.index).intersection(rg_cv.index)
    raw_cv, knn_cv, hgb_cv, rg_cv = (
        raw_cv[common], knn_cv[common], hgb_cv[common], rg_cv[common])

    # Bootstrap CI
    rng = np.random.default_rng(SEED)
    def bootstrap_reduction(base, model, n=1000):
        raw_med = base.median()
        reductions = []
        for _ in range(n):
            idx = rng.integers(0, len(base), len(base))
            reductions.append(1 - model.iloc[idx].median() / base.iloc[idx].median())
        reductions = sorted(reductions)
        return (float(100 * (1 - model.median() / raw_med)),
                float(100 * reductions[25]), float(100 * reductions[975]))

    raw_med = float(100 * raw_cv.median())
    knn_med, knn_lo, knn_hi = bootstrap_reduction(raw_cv, knn_cv)
    hgb_med, hgb_lo, hgb_hi = bootstrap_reduction(raw_cv, hgb_cv)
    rg_med, rg_lo, rg_hi = bootstrap_reduction(raw_cv, rg_cv)

    # Load NSF result from value_benchmark on the same split
    nsf_note = ""
    nsf_row = ""
    vb_csv = Path("eval/value_benchmark_results/per_star_cv.csv")
    vb_full_csv = Path("eval/value_benchmark_full_results/per_star_cv.csv")
    for label, path in [("Compact NSF", vb_csv), ("Full NSF", vb_full_csv)]:
        if path.exists():
            vb = pd.read_csv(path, index_col=0)
            if "Fresh compact NSF / 24 inputs" in vb.columns or "Fresh full NSF / 24 inputs" in vb.columns:
                col = ([c for c in vb.columns if "NSF" in c] or [None])[0]
                if col:
                    nsf_common = vb.loc[vb.index.isin(common.astype(float))]
                    nsf_cv = nsf_common[col]
                    raw_nsf = vb.loc[vb.index.isin(common.astype(float)), "Raw"]
                    nsf_m, nsf_lo2, nsf_hi2 = bootstrap_reduction(raw_nsf, nsf_cv)
                    nsf_row += (f"| {label} (value_benchmark split) | "
                                f"{len(nsf_common):,} | "
                                f"{float(100*nsf_cv.median()):.3f} | "
                                f"{nsf_m:.1f} ({nsf_lo2:.1f}, {nsf_hi2:.1f}) | "
                                f"{float(100*(nsf_cv>raw_nsf).mean()):.1f} |\n")
                    nsf_note = " NSF rows reuse saved predictions from value_benchmark_results."

    elapsed = time.time() - t0
    n_stars = len(common)

    report = f"""# Reconciled benchmark

All models evaluated on the **identical** value_benchmark canonical split.
- Split seed: {SEED}
- Cohort: tmag ≤ {MAX_TMAG}, sector_median > 0, gaiarp non-null, ≥ {MIN_SECTORS} sectors
- Features: spatial KNN from reference + train stars (n_ref+train = {len(ref_for_features):,})
- Metric: per-star coefficient of variation (corrected flux, no clipping)
- Bootstrap: 1,000 resamples over test stars
{nsf_note}

| Model | N stars | Median CV % | Reduction % (95% CI) | Stars worse % |
|---|---:|---:|---:|---:|
| Raw | {n_stars:,} | {raw_med:.3f} | — | — |
| Val-calibrated KNN | {n_stars:,} | {float(100*knn_cv.median()):.3f} | {knn_med:.1f} ({knn_lo:.1f}, {knn_hi:.1f}) | {float(100*(knn_cv>raw_cv).mean()):.1f} |
| Residual HGB (leaves={best_leaves}) | {n_stars:,} | {float(100*hgb_cv.median()):.3f} | {hgb_med:.1f} ({hgb_lo:.1f}, {hgb_hi:.1f}) | {float(100*(hgb_cv>raw_cv).mean()):.1f} |
| Robust global HGB | {n_stars:,} | {float(100*rg_cv.median()):.3f} | {rg_med:.1f} ({rg_lo:.1f}, {rg_hi:.1f}) | {float(100*(rg_cv>raw_cv).mean()):.1f} |
{nsf_row}
Notes:
- Robust global trains on log(F) − log(KNN) with alternating per-star median intercepts (L1 loss, {5} rounds).
- Residual HGB trains on LOO − KNN with L1 loss; leaf count tuned on validation set.
- KNN is 20-nearest-neighbour sector/CCD-matched median; alpha-shrinkage tuned on validation.
- Elapsed: {elapsed:.0f}s

Reproduce:
    python3 eval/reconciled_benchmark.py
"""

    (OUT / "REPORT.md").write_text(report)
    per = pd.DataFrame({"raw": raw_cv, "knn_calibrated": knn_cv,
                        "residual_hgb": hgb_cv, "robust_global": rg_cv})
    per.to_csv(OUT / "per_star_cv.csv")

    payload = {
        "n_common_stars": n_stars,
        "raw_median_cv_pct": raw_med,
        "knn_alpha": best_alpha_knn[0],
        "knn_median_cv_pct": float(100 * knn_cv.median()),
        "knn_reduction_pct": knn_med,
        "knn_ci95": [knn_lo, knn_hi],
        "hgb_best_leaves": best_leaves,
        "hgb_alpha": alpha_hgb,
        "hgb_median_cv_pct": float(100 * hgb_cv.median()),
        "hgb_reduction_pct": hgb_med,
        "hgb_ci95": [hgb_lo, hgb_hi],
        "robust_global_beta": beta,
        "robust_global_median_cv_pct": float(100 * rg_cv.median()),
        "robust_global_reduction_pct": rg_med,
        "robust_global_ci95": [rg_lo, rg_hi],
        "elapsed_seconds": elapsed,
    }
    (OUT / "results.json").write_text(json.dumps(payload, indent=2))
    print(report, flush=True)


if __name__ == "__main__":
    main()
