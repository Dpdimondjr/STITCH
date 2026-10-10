"""Held-out sector validation: train on sectors 1-55, evaluate on 56-98.

Tests whether model improvements survive a genuine temporal/spatial
extrapolation — new sky footprints the model has never seen — rather than
interpolating between already-observed detector positions.

Compares three models:
  1. Cross-sector KNN median (sector-agnostic spatial lookup)
  2. Residual HGB
  3. Robust global HGB

Baseline comparison: random-TIC split result from reconciled_benchmark.

Run from the repository root:
    python3 eval/held_out_sector_validation.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent))

OUT = Path("eval/held_out_sector_results")
SEED = 20260915
TRAIN_SECTORS = (1, 55)
TEST_SECTORS = (56, 98)
MIN_TRAIN_SECTORS = 5   # min early-period sectors for a training star
MIN_TEST_SECTORS = 5    # min late-period sectors for evaluation
MAX_TMAG = 13
REF_CLIP = (0.5, 1.5)
BASE_COLS = ["col", "row", "delta_sub_col", "delta_sub_row", "sector", "cam", "ccd",
             "tmag", "crowdsap", "cdpp1_0", "pdcvar", "jitter_rms",
             "pdc_noi", "pr_wght2", "gaiarp"]


# ── metric (same as reconciled_benchmark) ────────────────────────────────────

def cv_per_star(frame, offsets, min_obs=5):
    work = frame[["tic_id", "sector_median"]].copy()
    work["corrected"] = work.sector_median.to_numpy() / offsets
    g = work.groupby("tic_id").corrected
    cv = g.std(ddof=0) / g.mean()
    return cv[work.groupby("tic_id").size() >= min_obs]


# ── cross-sector KNN (CCD-only, sector-agnostic) ──────────────────────────────

def cross_sector_knn_features(ref_stars, query, k=20):
    """Build KNN features using reference stars' median position per CCD.

    ref_stars: one row per (tic_id, cam, ccd) with columns col, row, mean_loo.
    query: per-observation rows with col, row, cam, ccd, tic_id.
    Returns array of shape (len(query), 4): [knn_mean, knn_median, knn_mad, knn_radius].
    """
    query = query.reset_index(drop=True)
    out = np.full((len(query), 4), np.nan)
    for key, qg in query.groupby(["cam", "ccd"]):
        r = ref_stars[(ref_stars.cam == key[0]) & (ref_stars.ccd == key[1])]
        if len(r) < 2:
            continue
        coords = r[["col", "row"]].to_numpy(float)
        qcoords = qg[["col", "row"]].to_numpy(float)
        qids = qg.tic_id.to_numpy()
        rids = r.tic_id.to_numpy()
        rloo = r.mean_loo.to_numpy()
        take = min(k + int(r.tic_id.value_counts().max()) + 1, len(r))
        dists, idxs = cKDTree(coords).query(qcoords, k=take)
        if take == 1:
            dists, idxs = dists[:, None], idxs[:, None]
        for i, (q_id, nn, dd) in enumerate(zip(qids, idxs, dists)):
            keep = rids[nn] != q_id
            nn_k, dd_k = nn[keep][:k], dd[keep][:k]
            if len(nn_k) < 1:
                continue
            vals = rloo[nn_k]
            med = np.median(vals)
            out[qg.index[i]] = [vals.mean(), med,
                                 np.median(np.abs(vals - med)),
                                 dd_k[-1] if len(dd_k) else np.nan]
    return pd.DataFrame(out, columns=["knn_mean", "knn_median", "knn_mad", "knn_radius"],
                        index=query.index)


# ── robust global model (same as reconciled_benchmark) ───────────────────────

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


def tune_beta(frame, log_base, residual):
    best = None
    for beta in np.linspace(-0.25, 1.5, 36):
        pred = np.exp(log_base + beta * residual)
        score = float(cv_per_star(frame, pred).median())
        if best is None or score < best[1]:
            best = (float(beta), score)
    return best


def tune_alpha(frame, raw_pred):
    best = None
    for alpha in np.linspace(0, 1.25, 26):
        pred = 1 + alpha * (raw_pred - 1)
        score = float(cv_per_star(frame, pred).median())
        if best is None or score < best[1]:
            best = (float(alpha), score)
    return best


# ── bootstrap CI ─────────────────────────────────────────────────────────────

def bootstrap_reduction(raw_cv, model_cv, n=1000):
    rng = np.random.default_rng(SEED)
    raw_med = raw_cv.median()
    reductions = []
    for _ in range(n):
        idx = rng.integers(0, len(raw_cv), len(raw_cv))
        reductions.append(1 - model_cv.iloc[idx].median() / raw_cv.iloc[idx].median())
    reductions = sorted(reductions)
    return (float(100 * (1 - model_cv.median() / raw_med)),
            float(100 * reductions[25]),
            float(100 * reductions[975]))


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading parquet …", flush=True)
    df = pd.read_parquet("training_data_v3.parquet")
    valid = (df.sector.between(1, 100) & df.cam.isin([1,2,3,4]) &
             df.ccd.isin([1,2,3,4]) & df.tmag.le(MAX_TMAG) &
             df.sector_median.gt(0) & df.col.between(0, 2200) &
             df.row.between(0, 2200) & df.gaiarp.notna())
    df = df[valid].copy().replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["sector_median", "gaiarp"])

    # Recompute LOO labels in-sample (using all sectors for each star)
    g = df.groupby("tic_id").sector_median
    total, n = g.transform("sum"), g.transform("count")
    df["loo"] = df.sector_median / ((total - df.sector_median) / (n - 1))

    # Sector split
    train_obs = df[df.sector.between(*TRAIN_SECTORS)].copy()
    test_obs = df[df.sector.between(*TEST_SECTORS)].copy()

    # Stars with enough observations in each period
    train_counts = train_obs.groupby("tic_id").sector.nunique()
    test_counts = test_obs.groupby("tic_id").sector.nunique()
    train_eligible = set(train_counts[train_counts >= MIN_TRAIN_SECTORS].index)
    test_eligible = set(test_counts[test_counts >= MIN_TEST_SECTORS].index)
    eval_stars = train_eligible & test_eligible

    train_obs = train_obs[train_obs.tic_id.isin(train_eligible)].copy().reset_index(drop=True)
    test_obs = test_obs[test_obs.tic_id.isin(eval_stars)].copy().reset_index(drop=True)

    print(f"Training observations (sectors {TRAIN_SECTORS[0]}-{TRAIN_SECTORS[1]}): "
          f"{len(train_obs):,} rows, {train_obs.tic_id.nunique():,} stars", flush=True)
    print(f"Test observations (sectors {TEST_SECTORS[0]}-{TEST_SECTORS[1]}): "
          f"{len(test_obs):,} rows, {test_obs.tic_id.nunique():,} stars", flush=True)

    # Sub-split training stars into train/val for tuning (still within sectors 1-55)
    all_train_tics = np.array(sorted(train_obs.tic_id.unique()))
    val_size = max(500, len(all_train_tics) // 8)
    rng = np.random.default_rng(SEED)
    val_tics = set(rng.choice(all_train_tics, val_size, replace=False))
    fit_tics = set(all_train_tics) - val_tics
    fit_obs = train_obs[train_obs.tic_id.isin(fit_tics)].copy().reset_index(drop=True)
    val_obs = train_obs[train_obs.tic_id.isin(val_tics)].copy().reset_index(drop=True)
    print(f"Fit: {len(fit_tics):,} stars, Val: {len(val_tics):,} stars", flush=True)

    # Reference catalog: per-star mean position and mean LOO per CCD
    ref_stars = (train_obs.groupby(["tic_id", "cam", "ccd"])
                 .agg(col=("col", "median"), row=("row", "median"),
                      mean_loo=("loo", "mean"))
                 .reset_index())
    raw_loo = ref_stars.mean_loo.to_numpy()
    ref_stars["mean_loo"] = raw_loo.clip(*REF_CLIP)
    print(f"Reference catalog: {len(ref_stars):,} entries, "
          f"{ref_stars.tic_id.nunique():,} unique stars", flush=True)

    # Build cross-sector KNN features
    print("Building KNN features …", flush=True)
    feats_fit = cross_sector_knn_features(ref_stars, fit_obs)
    feats_val = cross_sector_knn_features(ref_stars, val_obs)
    feats_test = cross_sector_knn_features(ref_stars, test_obs)

    def make_X(frame, feats):
        base = frame[BASE_COLS].reset_index(drop=True).astype(float)
        return pd.concat([base, feats.reset_index(drop=True)], axis=1)

    X_fit = make_X(fit_obs, feats_fit)
    X_val = make_X(val_obs, feats_val)
    X_test = make_X(test_obs, feats_test)

    knn_fit = feats_fit.knn_median.to_numpy()
    knn_val_raw = feats_val.knn_median.to_numpy()
    knn_test = feats_test.knn_median.to_numpy()

    # Fallback: replace NaN KNN with 1.0
    knn_fit = np.where(np.isfinite(knn_fit), knn_fit, 1.0)
    knn_val_raw = np.where(np.isfinite(knn_val_raw), knn_val_raw, 1.0)
    knn_test = np.where(np.isfinite(knn_test), knn_test, 1.0)

    # ── KNN baseline ──
    print("KNN baseline …", flush=True)
    best_knn_alpha = None
    for alpha in np.linspace(0, 1.25, 26):
        pred = 1 + alpha * (knn_val_raw - 1)
        score = float(cv_per_star(val_obs, pred).median())
        if best_knn_alpha is None or score < best_knn_alpha[1]:
            best_knn_alpha = (float(alpha), score)
    knn_test_calibrated = 1 + best_knn_alpha[0] * (knn_test - 1)

    # ── Residual HGB ──
    print("Residual HGB …", flush=True)
    hgb_target = fit_obs.loo.to_numpy() - knn_fit
    best_hgb = None
    for leaves in [15, 31, 63]:
        m = HistGradientBoostingRegressor(
            loss="absolute_error", learning_rate=0.06, max_iter=250,
            max_leaf_nodes=leaves, min_samples_leaf=40, l2_regularization=0.1,
            random_state=SEED, early_stopping=False)
        m.fit(X_fit, hgb_target)
        raw_val = knn_val_raw + m.predict(X_val)
        score = float(cv_per_star(val_obs, raw_val).median())
        if best_hgb is None or score < best_hgb[2]:
            best_hgb = (m, leaves, score)
    res_hgb, best_leaves, _ = best_hgb
    raw_hgb_test = knn_test + res_hgb.predict(X_test)
    alpha_hgb, _ = tune_alpha(val_obs, knn_val_raw + res_hgb.predict(X_val))
    hgb_test = 1 + alpha_hgb * (raw_hgb_test - 1)
    print(f"  Residual HGB best leaves={best_leaves}, alpha={alpha_hgb:.3f}", flush=True)

    # ── Robust global ──
    print("Robust global model …", flush=True)
    log_flux_fit = np.log(fit_obs.sector_median.to_numpy())
    log_base_fit = np.log(np.clip(knn_fit, 1e-6, None))
    robust_model = fit_robust_global(X_fit, log_flux_fit, log_base_fit,
                                     fit_obs.tic_id.to_numpy())
    rv = robust_model.predict(X_val)
    beta, _ = tune_beta(val_obs, np.log(np.clip(knn_val_raw, 1e-6, None)), rv)
    robust_test = np.exp(np.log(np.clip(knn_test, 1e-6, None))
                         + beta * robust_model.predict(X_test))
    print(f"  Robust global beta={beta:.3f}", flush=True)

    # ── Evaluate on test period ──
    raw_cv = cv_per_star(test_obs, np.ones(len(test_obs)))
    knn_cv = cv_per_star(test_obs, knn_test_calibrated)
    hgb_cv = cv_per_star(test_obs, hgb_test)
    rg_cv = cv_per_star(test_obs, robust_test)
    common = raw_cv.index.intersection(knn_cv.index).intersection(
        hgb_cv.index).intersection(rg_cv.index)
    raw_cv, knn_cv, hgb_cv, rg_cv = (
        raw_cv[common], knn_cv[common], hgb_cv[common], rg_cv[common])

    raw_med = float(100 * raw_cv.median())
    knn_m, knn_lo, knn_hi = bootstrap_reduction(raw_cv, knn_cv)
    hgb_m, hgb_lo, hgb_hi = bootstrap_reduction(raw_cv, hgb_cv)
    rg_m, rg_lo, rg_hi = bootstrap_reduction(raw_cv, rg_cv)
    n_stars = len(common)
    elapsed = time.time() - t0

    report = f"""# Held-out sector validation

Train: sectors {TRAIN_SECTORS[0]}–{TRAIN_SECTORS[1]} | Test: sectors {TEST_SECTORS[0]}–{TEST_SECTORS[1]}

Tests whether corrections generalise to new sky footprints and epochs
rather than interpolating between already-seen detector positions.

- Training observations: {len(train_obs):,} rows, {train_obs.tic_id.nunique():,} stars
- Test observations: {len(test_obs):,} rows, {test_obs.tic_id.nunique():,} stars
  (stars with ≥{MIN_TEST_SECTORS} test sectors and ≥{MIN_TRAIN_SECTORS} training sectors)
- KNN features: cross-sector, cam/CCD-matched only (no sector key)
- Metric: per-star CV (corrected flux, no clipping)
- Bootstrap: 1,000 resamples over test stars

| Model | N stars | Median CV % | Reduction % (95% CI) | Stars worse % |
|---|---:|---:|---:|---:|
| Raw (sectors {TEST_SECTORS[0]}–{TEST_SECTORS[1]}) | {n_stars:,} | {raw_med:.3f} | — | — |
| Cross-sector KNN | {n_stars:,} | {float(100*knn_cv.median()):.3f} | {knn_m:.1f} ({knn_lo:.1f}, {knn_hi:.1f}) | {float(100*(knn_cv>raw_cv).mean()):.1f} |
| Residual HGB (leaves={best_leaves}) | {n_stars:,} | {float(100*hgb_cv.median()):.3f} | {hgb_m:.1f} ({hgb_lo:.1f}, {hgb_hi:.1f}) | {float(100*(hgb_cv>raw_cv).mean()):.1f} |
| Robust global HGB | {n_stars:,} | {float(100*rg_cv.median()):.3f} | {rg_m:.1f} ({rg_lo:.1f}, {rg_hi:.1f}) | {float(100*(rg_cv>raw_cv).mean()):.1f} |

Comparison baseline (random-TIC split, reconciled_benchmark.py):
| Raw | 1.447% | — |
| Cross-sector KNN | 1.012% | 30.1% (28.6, 31.3) |
| Residual HGB | 0.949% | 34.4% (33.0, 35.7) |
| Robust global | 0.931% | 35.7% (34.3, 36.9) |

Interpretation: if held-out sector reductions are substantially lower than the
random-TIC baseline, the current model gains are interpolation rather than
generalisation.

Elapsed: {elapsed:.0f}s

Reproduce:
    python3 eval/held_out_sector_validation.py
"""

    (OUT / "REPORT.md").write_text(report)
    per = pd.DataFrame({"raw": raw_cv, "knn": knn_cv,
                        "residual_hgb": hgb_cv, "robust_global": rg_cv})
    per.to_csv(OUT / "per_star_cv.csv")

    payload = {
        "train_sectors": list(TRAIN_SECTORS),
        "test_sectors": list(TEST_SECTORS),
        "n_train_stars": int(train_obs.tic_id.nunique()),
        "n_test_stars": int(test_obs.tic_id.nunique()),
        "n_common_eval_stars": n_stars,
        "raw_median_cv_pct": raw_med,
        "knn_alpha": best_knn_alpha[0],
        "knn_median_cv_pct": float(100 * knn_cv.median()),
        "knn_reduction_pct": knn_m,
        "knn_ci95": [knn_lo, knn_hi],
        "hgb_best_leaves": best_leaves,
        "hgb_alpha": alpha_hgb,
        "hgb_median_cv_pct": float(100 * hgb_cv.median()),
        "hgb_reduction_pct": hgb_m,
        "hgb_ci95": [hgb_lo, hgb_hi],
        "robust_global_beta": beta,
        "robust_global_median_cv_pct": float(100 * rg_cv.median()),
        "robust_global_reduction_pct": rg_m,
        "robust_global_ci95": [rg_lo, rg_hi],
        "elapsed_seconds": elapsed,
    }
    (OUT / "results.json").write_text(json.dumps(payload, indent=2))
    print(report, flush=True)


if __name__ == "__main__":
    main()
