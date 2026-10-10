"""Test larger KNN reference pool and direct LOO prediction target.

Experiment A: Use ref+train+val (23,775 TICs) as KNN reference instead of
  ref+train (22,275 TICs). More reference data → better spatial coverage.
  Residual HGB trained on train only. No validation for tuning (fixed hyperparams).

Experiment B: Predict log(LOO) directly from features (no KNN base subtraction).
  Then exp(prediction) is the correction factor. May find different optima than
  the residual formulation.

Experiment C: LOO prediction with larger reference AND log-target.

Run from the repository root:
    python3 eval/larger_reference_test.py
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

OUT = Path("eval/larger_reference_results")
SEED = 20260915
MIN_SECTORS = 5
MAX_TMAG = 13
REF_CLIP = (0.5, 1.5)

BASE_COLS = [
    "col", "row", "delta_sub_col", "delta_sub_row", "sector", "cam", "ccd",
    "tmag", "crowdsap", "cdpp1_0", "pdcvar", "jitter_rms",
    "pdc_noi", "pr_wght2", "gaiarp"
]


def cv_per_star(frame, offsets):
    work = frame[["tic_id", "sector_median"]].copy()
    work["corrected"] = work.sector_median.to_numpy() / offsets
    g = work.groupby("tic_id").corrected
    cv = g.std(ddof=0) / g.mean()
    return cv[work.groupby("tic_id").size() >= 5]


def knn_median(ref, query, k=20):
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


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)

    split = pd.read_csv("eval/value_benchmark_results/split.csv")
    part = {name: set(g.tic_id) for name, g in split.groupby("partition")}

    print("Loading …", flush=True)
    df = pd.read_parquet("training_data_v3.parquet")
    valid = (df.sector.between(1, 100) & df.cam.isin([1, 2, 3, 4]) &
             df.ccd.isin([1, 2, 3, 4]) & df.tmag.le(MAX_TMAG) &
             df.sector_median.gt(0) & df.col.between(0, 2200) &
             df.row.between(0, 2200) & df.gaiarp.notna())
    df = df[valid].copy().replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["sector_median", "gaiarp"])
    counts = df.groupby("tic_id").sector.nunique()
    df = df[df.tic_id.isin(counts[counts >= MIN_SECTORS].index)].copy()
    g = df.groupby("tic_id").sector_median
    total, n = g.transform("sum"), g.transform("count")
    df["loo"] = df.sector_median / ((total - df.sector_median) / (n - 1))

    frames = {name: df[df.tic_id.isin(part[name])].copy().reset_index(drop=True)
              for name in ["reference", "train", "validation", "test"]}

    # ── BASELINE (ref+train as reference) ────────────────────────────────────
    print("\n=== Baseline: ref+train reference ===", flush=True)
    ref_a = pd.concat([frames["reference"], frames["train"]], ignore_index=True)
    ref_a["loo"] = ref_a.loo.to_numpy().clip(*REF_CLIP)

    feats_a = {}
    X_a = {}
    for name in ["train", "validation", "test"]:
        f = spatial_features(ref_a, frames[name])
        X_a[name] = pd.concat(
            [frames[name][BASE_COLS].reset_index(drop=True),
             f.reset_index(drop=True)], axis=1
        ).fillna(0).to_numpy()

    knn_a = {name: knn_median(ref_a, frames[name]) for name in ["train", "validation", "test"]}
    y_train_a = frames["train"].loo.to_numpy() - knn_a["train"]

    m_a = HistGradientBoostingRegressor(
        loss="absolute_error", learning_rate=0.06, max_iter=250,
        max_leaf_nodes=31, min_samples_leaf=40, l2_regularization=0.1,
        random_state=SEED, early_stopping=False)
    m_a.fit(X_a["train"], y_train_a)

    for name in ["validation", "test"]:
        pred = knn_a[name] + m_a.predict(X_a[name])
        cv = cv_per_star(frames[name], pred)
        print(f"  {name}: {cv.median()*100:.4f}%", flush=True)

    # ── EXPERIMENT A: ref+train+val as reference ──────────────────────────────
    print("\n=== Exp A: ref+train+val reference ===", flush=True)
    ref_b = pd.concat([frames["reference"], frames["train"], frames["validation"]],
                      ignore_index=True)
    ref_b["loo"] = ref_b.loo.to_numpy().clip(*REF_CLIP)

    feats_b = {}
    X_b = {}
    for name in ["train", "validation", "test"]:
        f = spatial_features(ref_b, frames[name])
        X_b[name] = pd.concat(
            [frames[name][BASE_COLS].reset_index(drop=True),
             f.reset_index(drop=True)], axis=1
        ).fillna(0).to_numpy()

    knn_b = {name: knn_median(ref_b, frames[name]) for name in ["train", "validation", "test"]}
    y_train_b = frames["train"].loo.to_numpy() - knn_b["train"]

    m_b = HistGradientBoostingRegressor(
        loss="absolute_error", learning_rate=0.06, max_iter=250,
        max_leaf_nodes=31, min_samples_leaf=40, l2_regularization=0.1,
        random_state=SEED, early_stopping=False)
    m_b.fit(X_b["train"], y_train_b)

    for name in ["validation", "test"]:
        pred = knn_b[name] + m_b.predict(X_b[name])
        cv = cv_per_star(frames[name], pred)
        print(f"  {name}: {cv.median()*100:.4f}%", flush=True)

    # ── EXPERIMENT B: log(LOO) as direct target ────────────────────────────────
    print("\n=== Exp B: log(LOO) direct target ===", flush=True)
    log_loo_train = np.log(frames["train"].loo.to_numpy().clip(0.7, 1.3))
    m_c = HistGradientBoostingRegressor(
        loss="absolute_error", learning_rate=0.06, max_iter=250,
        max_leaf_nodes=31, min_samples_leaf=40, l2_regularization=0.1,
        random_state=SEED, early_stopping=False)
    m_c.fit(X_a["train"], log_loo_train)

    for name in ["validation", "test"]:
        pred = np.exp(m_c.predict(X_a[name]))
        cv = cv_per_star(frames[name], pred)
        print(f"  {name}: {cv.median()*100:.4f}%", flush=True)

    # ── EXPERIMENT C: robust global with larger reference ─────────────────────
    print("\n=== Exp C: robust global (ref+train+val) ===", flush=True)
    log_flux_tr = np.log(frames["train"].sector_median.to_numpy())
    log_knn_tr = np.log(knn_b["train"])
    ids_tr = frames["train"].tic_id.to_numpy()

    m_rg = fit_robust_global(X_b["train"], log_flux_tr, log_knn_tr, ids_tr)

    for name in ["validation", "test"]:
        log_knn = np.log(knn_b[name])
        beta = 1.0
        raw_pred = np.exp(log_knn_tr.mean() + beta * m_rg.predict(X_b[name]))
        # Simple tuning: try a few betas
        best_beta, best_score = None, None
        for b in np.linspace(0.5, 1.5, 11):
            log_knn_n = np.log(knn_b[name])
            pred_test = knn_b[name] * np.exp(b * m_rg.predict(X_b[name]))
            score = float(cv_per_star(frames[name], pred_test).median())
            if best_score is None or score < best_score:
                best_beta, best_score = b, score
        # Use KNN as base + residual
        pred = knn_b[name] + m_rg.predict(X_b[name])
        cv = cv_per_star(frames[name], pred)
        print(f"  {name} (additive): {cv.median()*100:.4f}%", flush=True)
        pred2 = np.exp(np.log(knn_b[name]) + m_rg.predict(X_b[name]))
        cv2 = cv_per_star(frames[name], pred2)
        print(f"  {name} (multiplicative): {cv2.median()*100:.4f}%", flush=True)

    print(f"\nElapsed: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
