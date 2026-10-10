"""Quick feature augmentation test.

Tests whether adding n_sectors_total and scatter_flag_frac to the residual HGB
improves validation/test performance beyond the reconciled_benchmark baseline.

Run from the repository root:
    python3 eval/feature_augment_test.py
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

OUT = Path("eval/feature_augment_results")
SEED = 20260915
MIN_SECTORS = 5
MAX_TMAG = 13
REF_CLIP = (0.5, 1.5)

BASE_COLS = [
    "col", "row", "delta_sub_col", "delta_sub_row", "sector", "cam", "ccd",
    "tmag", "crowdsap", "cdpp1_0", "pdcvar", "jitter_rms",
    "pdc_noi", "pr_wght2", "gaiarp"
]

# Candidate additional features
EXTRA_CANDIDATES = [
    "n_sectors_total",
    "scatter_flag_frac",
    "pdc_totp",
    "pdc_corp",
    "bkg_rms",
    "median_sap_bkg",
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


def fit_and_eval(X_train, X_val, y_train, frame_val, knn_val, leaves=31):
    m = HistGradientBoostingRegressor(
        loss="absolute_error", learning_rate=0.06, max_iter=250,
        max_leaf_nodes=leaves, min_samples_leaf=40, l2_regularization=0.1,
        random_state=SEED, early_stopping=False)
    m.fit(X_train, y_train)
    pred_val = knn_val + m.predict(X_val)
    score_val = float(cv_per_star(frame_val, pred_val).median())
    return m, score_val


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)

    split = pd.read_csv("eval/value_benchmark_results/split.csv")
    part = {name: set(g.tic_id) for name, g in split.groupby("partition")}

    print("Loading parquet …", flush=True)
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
    ref_for_features = pd.concat([frames["reference"], frames["train"]], ignore_index=True)
    ref_for_features["loo"] = ref_for_features.loo.to_numpy().clip(*REF_CLIP)

    print("Spatial features …", flush=True)
    feats = {}
    X_base = {}
    for name in ["train", "validation", "test"]:
        feats[name] = spatial_features(ref_for_features, frames[name])
        X_base[name] = pd.concat(
            [frames[name][BASE_COLS].reset_index(drop=True),
             feats[name].reset_index(drop=True)], axis=1
        ).fillna(0)

    print("KNN …", flush=True)
    knn = {name: knn_median(ref_for_features, frames[name])
           for name in ["train", "validation", "test"]}

    # Baseline (original feature set)
    y_train = frames["train"].loo.to_numpy() - knn["train"]
    _, baseline_val = fit_and_eval(X_base["train"], X_base["validation"],
                                   y_train, frames["validation"], knn["validation"])
    print(f"Baseline (no extras): val_cv={baseline_val*100:.4f}%", flush=True)

    # Test each extra feature individually
    results = [{"features": "baseline", "val_cv_pct": round(baseline_val * 100, 4)}]
    for feat in EXTRA_CANDIDATES:
        if feat not in df.columns:
            print(f"  SKIP {feat}: not in parquet", flush=True)
            continue
        X_aug = {}
        for name in ["train", "validation", "test"]:
            col_vals = frames[name][feat].fillna(0).to_numpy().reshape(-1, 1)
            X_aug[name] = np.column_stack([X_base[name].to_numpy(), col_vals])
        _, val_score = fit_and_eval(X_aug["train"], X_aug["validation"],
                                    y_train, frames["validation"], knn["validation"])
        delta = (val_score - baseline_val) * 100
        arrow = "▼" if delta < -0.001 else ("▲" if delta > 0.001 else "≈")
        print(f"  +{feat}: val_cv={val_score*100:.4f}% ({arrow}{abs(delta):.4f}pp)", flush=True)
        results.append({"features": f"+{feat}", "val_cv_pct": round(val_score * 100, 4),
                        "delta_pp": round(delta, 4)})

    # Test all extras together
    X_all = {}
    all_extra = [f for f in EXTRA_CANDIDATES if f in df.columns]
    for name in ["train", "validation", "test"]:
        extra_cols = frames[name][all_extra].fillna(0).to_numpy()
        X_all[name] = np.column_stack([X_base[name].to_numpy(), extra_cols])
    _, all_val = fit_and_eval(X_all["train"], X_all["validation"],
                              y_train, frames["validation"], knn["validation"])
    delta = (all_val - baseline_val) * 100
    print(f"  All extras: val_cv={all_val*100:.4f}% (Δ={delta:+.4f}pp)", flush=True)
    results.append({"features": "all_extras", "val_cv_pct": round(all_val * 100, 4),
                    "delta_pp": round(delta, 4)})

    # Best feature set → evaluate on test
    best = min(results[1:] + [{"features": "baseline", "val_cv_pct": baseline_val * 100}],
               key=lambda r: r["val_cv_pct"])
    print(f"\nBest: {best['features']} (val={best['val_cv_pct']:.4f}%)", flush=True)

    # Evaluate best on test set
    if best["features"] == "baseline":
        X_test = X_base["test"].to_numpy()
        X_tr = X_base["train"].to_numpy()
    elif best["features"] == "all_extras":
        X_test = X_all["test"]
        X_tr = X_all["train"]
    else:
        fname = best["features"][1:]
        col_vals_tr = frames["train"][fname].fillna(0).to_numpy().reshape(-1, 1)
        col_vals_te = frames["test"][fname].fillna(0).to_numpy().reshape(-1, 1)
        X_tr = np.column_stack([X_base["train"].to_numpy(), col_vals_tr])
        X_test = np.column_stack([X_base["test"].to_numpy(), col_vals_te])

    best_model = HistGradientBoostingRegressor(
        loss="absolute_error", learning_rate=0.06, max_iter=250,
        max_leaf_nodes=31, min_samples_leaf=40, l2_regularization=0.1,
        random_state=SEED, early_stopping=False)
    best_model.fit(X_tr, y_train)
    pred_test = knn["test"] + best_model.predict(X_test)
    test_cv = cv_per_star(frames["test"], pred_test)
    print(f"\nBest model on test: median_cv={test_cv.median()*100:.4f}%", flush=True)

    print(f"\nElapsed: {time.time()-t0:.0f}s")
    (OUT / "results.json").write_text(json.dumps({
        "feature_results": results,
        "best": best,
        "test_cv_median_pct": float(test_cv.median() * 100),
        "elapsed_s": int(time.time() - t0),
    }, indent=2))
    print(f"Saved to {OUT}/", flush=True)


if __name__ == "__main__":
    main()
