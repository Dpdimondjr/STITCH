"""Test using ALL valid non-test TICs as KNN reference.

The benchmark split uses 22,275 (ref+train+val) TICs for spatial features, but
there are 37,789 valid non-test TICs in the full parquet. Using all of them
as KNN reference should improve spatial coverage significantly.

The residual HGB is still trained only on the benchmark train partition (6,000 TICs)
to preserve comparability. The gain comes purely from better KNN neighbors.

Run from the repository root:
    python3 eval/full_reference_test.py
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

OUT = Path("eval/full_reference_results")
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
    test_tics = part["test"]

    print("Loading full parquet …", flush=True)
    df = pd.read_parquet("training_data_v3.parquet")
    valid = (df.sector.between(1, 100) & df.cam.isin([1, 2, 3, 4]) &
             df.ccd.isin([1, 2, 3, 4]) & df.tmag.le(MAX_TMAG) &
             df.sector_median.gt(0) & df.col.between(0, 2200) &
             df.row.between(0, 2200) & df.gaiarp.notna())
    df = df[valid].copy().replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["sector_median", "gaiarp"])
    counts = df.groupby("tic_id").sector.nunique()
    df = df[df.tic_id.isin(counts[counts >= MIN_SECTORS].index)].copy()

    # Recompute LOO on full (non-test) dataset — use flux_offset_loo from parquet
    # for non-benchmark stars (already computed globally)
    g = df.groupby("tic_id").sector_median
    total, n = g.transform("sum"), g.transform("count")
    df["loo"] = df.sector_median / ((total - df.sector_median) / (n - 1))

    # All non-test valid TICs — expanded reference
    df_nonttest = df[~df.tic_id.isin(test_tics)].copy()
    print(f"Full reference: {df_nonttest.tic_id.nunique():,} TICs, {len(df_nonttest):,} rows", flush=True)

    # Benchmark partitions
    frames = {name: df[df.tic_id.isin(part[name])].copy().reset_index(drop=True)
              for name in ["reference", "train", "validation", "test"]}
    print(f"Benchmark: ref={frames['reference'].tic_id.nunique():,}, "
          f"train={frames['train'].tic_id.nunique():,}, "
          f"val={frames['validation'].tic_id.nunique():,}, "
          f"test={frames['test'].tic_id.nunique():,}", flush=True)

    # Expanded reference (all valid non-test TICs)
    ref_full = df_nonttest.copy()
    ref_full["loo"] = ref_full.loo.to_numpy().clip(*REF_CLIP)

    # For spatial_features, we need flux_offset_loo — use our recomputed loo
    ref_full["flux_offset_loo"] = ref_full["loo"]

    # Benchmark reference (ref+train) — same as reconciled_benchmark
    ref_small = pd.concat([frames["reference"], frames["train"]], ignore_index=True)
    ref_small["loo"] = ref_small.loo.to_numpy().clip(*REF_CLIP)
    ref_small["flux_offset_loo"] = ref_small["loo"]

    print("\n=== Baseline: ref+train (22,275 TICs) ===", flush=True)
    X_small = {}
    for name in ["train", "validation", "test"]:
        f = spatial_features(ref_small, frames[name])
        X_small[name] = pd.concat(
            [frames[name][BASE_COLS].reset_index(drop=True),
             f.reset_index(drop=True)], axis=1
        ).fillna(0).to_numpy()

    knn_small = {name: knn_median(ref_small, frames[name])
                 for name in ["train", "validation", "test"]}
    y_train = frames["train"].loo.to_numpy() - knn_small["train"]

    m_small = HistGradientBoostingRegressor(
        loss="absolute_error", learning_rate=0.06, max_iter=250,
        max_leaf_nodes=31, min_samples_leaf=40, l2_regularization=0.1,
        random_state=SEED, early_stopping=False)
    m_small.fit(X_small["train"], y_train)

    for name in ["validation", "test"]:
        pred = knn_small[name] + m_small.predict(X_small[name])
        cv = cv_per_star(frames[name], pred)
        print(f"  {name}: {cv.median()*100:.4f}%", flush=True)

    print(f"\n=== Expanded: all non-test ({ref_full.tic_id.nunique():,} TICs) ===", flush=True)
    X_full = {}
    for name in ["train", "validation", "test"]:
        f = spatial_features(ref_full, frames[name])
        X_full[name] = pd.concat(
            [frames[name][BASE_COLS].reset_index(drop=True),
             f.reset_index(drop=True)], axis=1
        ).fillna(0).to_numpy()

    knn_full = {name: knn_median(ref_full, frames[name])
                for name in ["train", "validation", "test"]}
    y_train_full = frames["train"].loo.to_numpy() - knn_full["train"]

    m_full = HistGradientBoostingRegressor(
        loss="absolute_error", learning_rate=0.06, max_iter=250,
        max_leaf_nodes=31, min_samples_leaf=40, l2_regularization=0.1,
        random_state=SEED, early_stopping=False)
    m_full.fit(X_full["train"], y_train_full)

    for name in ["validation", "test"]:
        pred = knn_full[name] + m_full.predict(X_full[name])
        cv = cv_per_star(frames[name], pred)
        print(f"  {name}: {cv.median()*100:.4f}%", flush=True)

    # Robust global with full reference
    print(f"\n=== Robust global (full reference) ===", flush=True)
    log_flux_tr = np.log(frames["train"].sector_median.to_numpy())
    log_knn_tr = np.log(knn_full["train"])
    ids_tr = frames["train"].tic_id.to_numpy()

    m_rg = fit_robust_global(X_full["train"], log_flux_tr, log_knn_tr, ids_tr)

    # Tune beta on validation
    best_beta, best_val = None, None
    for beta in np.linspace(-0.25, 1.5, 36):
        pred = knn_full["validation"] * np.exp(beta * m_rg.predict(X_full["validation"]))
        score = float(cv_per_star(frames["validation"], pred).median())
        if best_val is None or score < best_val:
            best_beta, best_val = beta, score
    print(f"  best beta={best_beta:.3f}, val_cv={best_val*100:.4f}%", flush=True)

    pred_test = knn_full["test"] * np.exp(best_beta * m_rg.predict(X_full["test"]))
    test_cv_rg = cv_per_star(frames["test"], pred_test)
    print(f"  test: {test_cv_rg.median()*100:.4f}%", flush=True)

    print(f"\nElapsed: {time.time()-t0:.0f}s")

    (OUT / "results.json").write_text(json.dumps({
        "ref_full_tics": int(ref_full.tic_id.nunique()),
        "elapsed_s": int(time.time() - t0),
    }, indent=2))


if __name__ == "__main__":
    main()
