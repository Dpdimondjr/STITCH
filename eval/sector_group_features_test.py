"""Test sector-level aggregate features and per-star LOO variance features.

Two new feature types:
1. Sector-group aggregates: mean/std of bkg, pdcvar, pdc_corp per (sector,cam,ccd) group.
   Captures group-level systematics (scattered light, CBV correction magnitude).
2. Per-star LOO std: leave-one-out std of a star's LOO values across sectors.
   Captures how variable a star is — model should be more conservative for high-std stars.
3. Combination of both.

Run from the repository root:
    python3 eval/sector_group_features_test.py
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

OUT = Path("eval/sector_group_results")
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


def make_sector_group_features(df_ref, df_query):
    """For each (sector,cam,ccd) group, compute aggregate stats from ref."""
    agg = df_ref.groupby(["sector", "cam", "ccd"]).agg(
        grp_bkg_mean=("median_sap_bkg", "mean"),
        grp_bkg_std=("median_sap_bkg", "std"),
        grp_pdcvar_mean=("pdcvar", "mean"),
        grp_pdc_corp_mean=("pdc_corp", "mean"),
        grp_scatter_frac=("scatter_flag_frac", "mean"),
        grp_loo_mean=("loo", "mean"),
        grp_loo_std=("loo", "std"),
        grp_n=("tic_id", "count"),
    ).reset_index()
    merged = df_query[["sector", "cam", "ccd"]].merge(agg, on=["sector", "cam", "ccd"], how="left")
    feat_cols = [c for c in agg.columns if c.startswith("grp_")]
    return merged[feat_cols].fillna(0)


def make_star_loo_features(df):
    """Per-star leave-one-sector-out LOO std and mean (computed from full data)."""
    # For each (tic_id, sector): std and mean of LOO for ALL OTHER sectors of that star
    g = df.groupby("tic_id")["loo"]
    loo_sum = g.transform("sum")
    loo_sum2 = df.loo.pow(2).groupby(df.tic_id).transform("sum")
    n = g.transform("count")
    # Leave-one-out mean of LOO values for this star
    loo_loo_mean = (loo_sum - df.loo) / (n - 1).clip(lower=1)
    # Leave-one-out variance
    loo_loo_var = ((loo_sum2 - df.loo.pow(2)) / (n - 1).clip(lower=1) - loo_loo_mean.pow(2)).clip(lower=0)
    loo_loo_std = loo_loo_var.pow(0.5)
    return pd.DataFrame({
        "star_loo_loo_mean": loo_loo_mean.to_numpy(),
        "star_loo_loo_std": loo_loo_std.to_numpy(),
        "star_n_sectors": n.to_numpy().astype(float),
    })


def fit_and_eval(X_tr, y_tr, X_va, frame_va, knn_va, label=""):
    m = HistGradientBoostingRegressor(
        loss="absolute_error", learning_rate=0.06, max_iter=250,
        max_leaf_nodes=31, min_samples_leaf=40, l2_regularization=0.1,
        random_state=SEED, early_stopping=False)
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
        X_base[name] = pd.concat(
            [frames[name][BASE_COLS].reset_index(drop=True),
             f.reset_index(drop=True)], axis=1
        ).fillna(0)

    print("KNN …", flush=True)
    knn = {name: knn_median(ref_for_features, frames[name])
           for name in ["train", "validation", "test"]}
    y_train = frames["train"].loo.to_numpy() - knn["train"]

    print("Group and star features …", flush=True)
    # Sector group features (computed from reference only — no leakage)
    grp = {name: make_sector_group_features(ref_for_features, frames[name])
           for name in ["train", "validation", "test"]}

    # Per-star LOO std (computed within each partition — NO cross-partition leakage)
    star_feat = {name: make_star_loo_features(frames[name])
                 for name in ["train", "validation", "test"]}

    print("\nExperiments:", flush=True)
    results = []

    # 0. Baseline
    _, s0 = fit_and_eval(X_base["train"].to_numpy(), y_train,
                         X_base["validation"].to_numpy(),
                         frames["validation"], knn["validation"], "baseline")
    results.append({"config": "baseline", "val_cv_pct": round(s0 * 100, 4)})

    # 1. +sector group features
    X_grp = {name: np.column_stack([X_base[name].to_numpy(), grp[name].to_numpy()])
             for name in ["train", "validation", "test"]}
    _, s1 = fit_and_eval(X_grp["train"], y_train, X_grp["validation"],
                         frames["validation"], knn["validation"], "+sector_group_feats")
    results.append({"config": "+sector_group", "val_cv_pct": round(s1 * 100, 4),
                    "delta_pp": round((s1 - s0) * 100, 4)})

    # 2. +star LOO std
    X_star = {name: np.column_stack([X_base[name].to_numpy(), star_feat[name].to_numpy()])
              for name in ["train", "validation", "test"]}
    _, s2 = fit_and_eval(X_star["train"], y_train, X_star["validation"],
                         frames["validation"], knn["validation"], "+star_loo_std")
    results.append({"config": "+star_loo_std", "val_cv_pct": round(s2 * 100, 4),
                    "delta_pp": round((s2 - s0) * 100, 4)})

    # 3. +both
    X_both = {name: np.column_stack([X_base[name].to_numpy(), grp[name].to_numpy(),
                                     star_feat[name].to_numpy()])
              for name in ["train", "validation", "test"]}
    _, s3 = fit_and_eval(X_both["train"], y_train, X_both["validation"],
                         frames["validation"], knn["validation"], "+group+star")
    results.append({"config": "+group+star", "val_cv_pct": round(s3 * 100, 4),
                    "delta_pp": round((s3 - s0) * 100, 4)})

    print("\n=== Summary ===")
    best = min(results, key=lambda r: r["val_cv_pct"])
    for r in results:
        marker = " ← best" if r["config"] == best["config"] else ""
        d = f" ({r['delta_pp']:+.4f}pp)" if "delta_pp" in r else ""
        print(f"  {r['config']:30s}: {r['val_cv_pct']:.4f}%{d}{marker}")

    # Correlations: do star_loo_std and group features correlate with LOO?
    print("\n=== Feature correlation with LOO (train set) ===")
    loo_tr = frames["train"].loo.to_numpy()
    for feat_name, feat_arr in [
        ("star_loo_loo_std", star_feat["train"]["star_loo_loo_std"].to_numpy()),
        ("star_n_sectors",   star_feat["train"]["star_n_sectors"].to_numpy()),
        ("grp_bkg_mean",     grp["train"]["grp_bkg_mean"].to_numpy()),
        ("grp_loo_std",      grp["train"]["grp_loo_std"].to_numpy()),
        ("grp_pdc_corp",     grp["train"]["grp_pdc_corp_mean"].to_numpy()),
    ]:
        r_val = np.corrcoef(feat_arr, loo_tr)[0, 1]
        r_abs = np.corrcoef(feat_arr, np.abs(loo_tr - 1))[0, 1]
        print(f"  r({feat_name}, LOO):     {r_val:.4f}")
        print(f"  r({feat_name}, |LOO-1|): {r_abs:.4f}")

    print(f"\nElapsed: {time.time()-t0:.0f}s")
    (OUT / "results.json").write_text(json.dumps({
        "results": results,
        "best": best,
        "elapsed_s": int(time.time() - t0),
    }, indent=2))
    print(f"Saved to {OUT}/", flush=True)


if __name__ == "__main__":
    main()
