"""Test whether treating cam/ccd as categorical features in HistGBM helps.

HistGBM with categorical_features uses a proper one-hot-like split for categorical
variables, which can capture camera-specific patterns that ordinal splits miss.
Also tests adding cam*ccd interaction as a single categorical.

Run from the repository root:
    python3 eval/categorical_cam_test.py
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

OUT = Path("eval/categorical_cam_results")
SEED = 20260915
MIN_SECTORS = 5
MAX_TMAG = 13
REF_CLIP = (0.5, 1.5)

BASE_COLS = [
    "col", "row", "delta_sub_col", "delta_sub_row", "sector", "cam", "ccd",
    "tmag", "crowdsap", "cdpp1_0", "pdcvar", "jitter_rms",
    "pdc_noi", "pr_wght2", "gaiarp"
]
CAM_IDX = BASE_COLS.index("cam")    # column index for cam
CCD_IDX = BASE_COLS.index("ccd")    # column index for ccd


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


def run(X_tr, y_tr, X_va, frame_va, knn_va, cat_feats=None, label=""):
    m = HistGradientBoostingRegressor(
        loss="absolute_error", learning_rate=0.06, max_iter=250,
        max_leaf_nodes=31, min_samples_leaf=40, l2_regularization=0.1,
        random_state=SEED, early_stopping=False,
        categorical_features=cat_feats)
    m.fit(X_tr, y_tr)
    pred = knn_va + m.predict(X_va)
    score = float(cv_per_star(frame_va, pred).median())
    print(f"  {label}: val_cv={score*100:.4f}%", flush=True)
    return m, score


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
    ref_for_features = pd.concat([frames["reference"], frames["train"]], ignore_index=True)
    ref_for_features["loo"] = ref_for_features.loo.to_numpy().clip(*REF_CLIP)

    print("Spatial features …", flush=True)
    X_base = {}
    for name in ["train", "validation", "test"]:
        f = spatial_features(ref_for_features, frames[name])
        X_base[name] = np.column_stack([
            frames[name][BASE_COLS].fillna(0).to_numpy(float),
            f.fillna(0).to_numpy(float)
        ])

    print("KNN …", flush=True)
    knn = {name: knn_median(ref_for_features, frames[name])
           for name in ["train", "validation", "test"]}
    y_train = frames["train"].loo.to_numpy() - knn["train"]

    n_base = X_base["train"].shape[1]
    print(f"Feature matrix: {n_base} columns\n", flush=True)

    results = []

    # 1. Baseline (all numeric, no categorical)
    m_base, s_base = run(X_base["train"], y_train, X_base["validation"],
                         frames["validation"], knn["validation"],
                         cat_feats=None, label="baseline (numeric cam/ccd)")
    results.append({"config": "numeric", "val_cv_pct": round(s_base * 100, 4)})

    # 2. cam and ccd as categorical
    cat_idx = [CAM_IDX, CCD_IDX]
    _, s_cat = run(X_base["train"], y_train, X_base["validation"],
                   frames["validation"], knn["validation"],
                   cat_feats=cat_idx, label="cam+ccd categorical")
    results.append({"config": "cam_ccd_cat", "val_cv_pct": round(s_cat * 100, 4),
                    "delta_pp": round((s_cat - s_base) * 100, 4)})

    # 3. Add cam_ccd combined (0..15) as extra categorical feature
    X_with_combo = {}
    for name in ["train", "validation", "test"]:
        cam_col = frames[name]["cam"].to_numpy().astype(int) - 1
        ccd_col = frames[name]["ccd"].to_numpy().astype(int) - 1
        combo = (cam_col * 4 + ccd_col).reshape(-1, 1).astype(float)
        X_with_combo[name] = np.column_stack([X_base[name], combo])
    cat_idx_combo = [CAM_IDX, CCD_IDX, n_base]  # last col is combo
    _, s_combo = run(X_with_combo["train"], y_train, X_with_combo["validation"],
                     frames["validation"], knn["validation"],
                     cat_feats=cat_idx_combo, label="cam+ccd+combo categorical")
    results.append({"config": "cam_ccd_combo_cat", "val_cv_pct": round(s_combo * 100, 4),
                    "delta_pp": round((s_combo - s_base) * 100, 4)})

    # 4. sector as categorical too (sector is 1-98, already in BASE_COLS)
    sect_idx = BASE_COLS.index("sector")
    cat_idx_all = [CAM_IDX, CCD_IDX, sect_idx]
    _, s_allcat = run(X_base["train"], y_train, X_base["validation"],
                      frames["validation"], knn["validation"],
                      cat_feats=cat_idx_all, label="cam+ccd+sector categorical")
    results.append({"config": "cam_ccd_sector_cat", "val_cv_pct": round(s_allcat * 100, 4),
                    "delta_pp": round((s_allcat - s_base) * 100, 4)})

    print("\n=== Summary ===")
    best = min(results, key=lambda r: r["val_cv_pct"])
    for r in results:
        marker = " ← best" if r["config"] == best["config"] else ""
        print(f"  {r['config']:30s}: {r['val_cv_pct']:.4f}%{marker}")

    # Evaluate best on test
    best_cfg = best["config"]
    if best_cfg == "cam_ccd_cat":
        X_tr, X_te = X_base["train"], X_base["test"]
        cat_use = [CAM_IDX, CCD_IDX]
    elif best_cfg == "cam_ccd_combo_cat":
        X_tr, X_te = X_with_combo["train"], X_with_combo["test"]
        cat_use = cat_idx_combo
    elif best_cfg == "cam_ccd_sector_cat":
        X_tr, X_te = X_base["train"], X_base["test"]
        cat_use = cat_idx_all
    else:
        X_tr, X_te = X_base["train"], X_base["test"]
        cat_use = None

    best_model = HistGradientBoostingRegressor(
        loss="absolute_error", learning_rate=0.06, max_iter=250,
        max_leaf_nodes=31, min_samples_leaf=40, l2_regularization=0.1,
        random_state=SEED, early_stopping=False, categorical_features=cat_use)
    best_model.fit(X_tr, y_train)
    pred_te = knn["test"] + best_model.predict(X_te)
    test_cv = cv_per_star(frames["test"], pred_te)
    print(f"\nBest ({best_cfg}) on test: {test_cv.median()*100:.4f}%")

    # Per-camera test breakdown
    td_meta = df.groupby("tic_id").agg(cam=("cam", lambda x: int(x.mode()[0]))).reset_index()
    test_df = frames["test"].copy()
    test_df["robust_global"] = pred_te
    tic_cv = test_df.groupby("tic_id").apply(
        lambda g: pd.Series({
            "raw_cv": g.sector_median.std(ddof=0) / g.sector_median.mean(),
            "corr_cv": (g.sector_median / g.robust_global).std(ddof=0) / (g.sector_median / g.robust_global).mean()
        })
    ).reset_index()
    tic_cv = tic_cv.merge(td_meta, on="tic_id", how="left")
    print("\nPer-camera (best model on test):")
    for cam in sorted(tic_cv.cam.dropna().unique()):
        sub = tic_cv[tic_cv.cam == cam]
        print(f"  cam{int(cam)}: median corr_cv={sub.corr_cv.median()*100:.3f}%  n={len(sub)}")

    print(f"\nElapsed: {time.time()-t0:.0f}s")
    (OUT / "results.json").write_text(json.dumps({
        "results": results,
        "best_config": best_cfg,
        "test_cv_median_pct": float(test_cv.median() * 100),
        "elapsed_s": int(time.time() - t0),
    }, indent=2))
    print(f"Saved to {OUT}/", flush=True)


if __name__ == "__main__":
    main()
