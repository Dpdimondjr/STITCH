"""Leakage-safe global calibration with nuisance star intercepts.

Fits log(F_is) = a_i + log(b_is) + g(c_is), where b is the K=20
training-reference median-KNN correction.  Alternating updates of a_i and g
optimize the within-star (equivalently, pairwise squared-difference) objective;
the nuisance intercepts are never features and are never used at inference.
"""
from pathlib import Path
import json, time
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from residual_multiscale_experiment import (split_stars, spatial_features,
                                            scatter_table, tune_alpha)

SEED = 20260916
OUT = Path("eval/pairwise_global_calibration_results")
BASE_COLS = [
    "col", "row", "delta_sub_col", "delta_sub_row", "sector", "cam", "ccd",
    "tmag", "crowdsap", "cdpp1_0", "pdcvar", "jitter_rms", "pdc_noi",
    "pr_wght2", "gaiarp", "focal_plane_temp",
]
FEATURE_COLS = BASE_COLS + [
    "knn20_median", "knn20_mad", "knn20_radius", "reference_count"
]


def star_center(values, ids):
    """Subtract per-star means without exposing star identity to the model."""
    s = pd.Series(values)
    return values - s.groupby(pd.Series(ids)).transform("mean").to_numpy()


def fit_global_model(X, log_flux, log_base, ids, rounds=5):
    """Backfit nuisance star intercepts and the shared calibration function."""
    response = log_flux - log_base
    intercept = pd.Series(response).groupby(pd.Series(ids)).transform("mean").to_numpy()
    history = []
    model = None
    for iteration in range(rounds):
        target = response - intercept
        model = HistGradientBoostingRegressor(
            loss="squared_error", learning_rate=.06, max_iter=250,
            max_leaf_nodes=31, min_samples_leaf=40, l2_regularization=.1,
            random_state=SEED + iteration, early_stopping=False,
        )
        model.fit(X, target)
        fitted = model.predict(X)
        # Exact least-squares update for each nuisance intercept.
        intercept = (pd.Series(response - fitted)
                     .groupby(pd.Series(ids)).transform("mean").to_numpy())
        centered_error = star_center(response - fitted, ids)
        history.append({
            "iteration": iteration + 1,
            "within_star_rmse_log": float(np.sqrt(np.mean(centered_error**2))),
            "within_star_mae_log": float(np.mean(np.abs(centered_error))),
        })
    return model, history


def tune_residual_strength(frame, log_base, residual):
    best = None
    for beta in np.linspace(-.25, 1.5, 36):
        pred = np.exp(log_base + beta * residual)
        score = float(scatter_table(frame, {"candidate": pred}).candidate.median())
        if best is None or score < best[1]:
            best = (float(beta), score)
    return best


def paired_bootstrap(per, a, b, seed=SEED, draws=5000):
    """CI for percent reduction in residual scatter of a relative to b."""
    rng = np.random.default_rng(seed)
    x, y = per[a].to_numpy(), per[b].to_numpy()
    vals = np.empty(draws)
    for i in range(draws):
        take = rng.integers(0, len(per), len(per))
        vals[i] = 100 * (1 - np.median(x[take]) / np.median(y[take]))
    return [float(v) for v in np.quantile(vals, [.025, .5, .975])]


def main():
    t0 = time.time(); OUT.mkdir(exist_ok=True)
    df = pd.read_parquet("training_data_v3.parquet")
    df = df.dropna(subset=["col", "row", "flux_offset_loo", "sector_median"])
    df = df[df.flux_offset_loo.between(.85, 1.15, inclusive="neither") &
            df.n_sectors_total.ge(4) & df.tmag.le(13) &
            df.sector_median.gt(0)].copy().reset_index(drop=True)
    train_ids, val_ids, test_ids = split_stars(df)
    masks = {"train": df.tic_id.isin(train_ids),
             "val": df.tic_id.isin(val_ids),
             "test": df.tic_id.isin(test_ids)}
    reference = df[masks["train"]].copy()
    feats = {}
    frames = {}
    for name, mask in masks.items():
        frames[name] = df[mask].reset_index(drop=True)
        print("spatial features", name, len(frames[name]), flush=True)
        feats[name] = spatial_features(reference, df[mask])

    X = {}
    good = {}
    for name in masks:
        X[name] = pd.concat([frames[name][BASE_COLS], feats[name]], axis=1)[FEATURE_COLS]
        good[name] = np.isfinite(feats[name].knn20_median.to_numpy())
        # K=20 has no defined correction in sparse detector groups. Restrict
        # every method to observations where that common baseline exists.
        if not good[name].all():
            frames[name] = frames[name].loc[good[name]].reset_index(drop=True)
            feats[name] = feats[name].loc[good[name]].reset_index(drop=True)
            X[name] = X[name].loc[good[name]].reset_index(drop=True)

    base_train = feats["train"].knn20_median.to_numpy()
    model, history = fit_global_model(
        X["train"], np.log(frames["train"].sector_median.to_numpy()),
        np.log(base_train), frames["train"].tic_id.to_numpy())

    base_val = feats["val"].knn20_median.to_numpy()
    residual_val = model.predict(X["val"])
    beta, val_scatter = tune_residual_strength(
        frames["val"], np.log(base_val), residual_val)

    base_test = feats["test"].knn20_median.to_numpy()
    residual_test = model.predict(X["test"])
    pairwise_pred = np.exp(np.log(base_test) + beta * residual_test)

    # Refit the prior leakage-safe linear-offset HGB on the identical supported
    # observations, rather than comparing CSVs that may differ in sparse rows.
    prior_model = HistGradientBoostingRegressor(
        loss="absolute_error", learning_rate=.06, max_iter=250,
        max_leaf_nodes=31, min_samples_leaf=40, l2_regularization=.1,
        random_state=20260915, early_stopping=False)
    prior_target = frames["train"].flux_offset_loo.to_numpy() - base_train
    prior_model.fit(X["train"], prior_target)
    prior_val_raw = base_val + prior_model.predict(X["val"])
    prior_alpha, prior_val_scatter = tune_alpha(frames["val"], prior_val_raw)
    prior_test_raw = base_test + prior_model.predict(X["test"])
    prior_pred = 1 + prior_alpha * (prior_test_raw - 1)
    per = scatter_table(frames["test"], {
        "raw": np.ones(len(frames["test"])),
        "median_knn": base_test,
        "pairwise_global": pairwise_pred,
        "residual_k20": prior_pred,
    })

    summary = {}
    for col in per:
        summary[col] = {
            "median_scatter_pct": float(100 * per[col].median()),
            "reduction_from_raw_pct": float(100 * (1 - per[col].median()/per.raw.median())),
            "stars_worse_than_raw_pct": float(100 * (per[col] > per.raw).mean()),
        }
    comparisons = {
        "pairwise_vs_median_knn_pct": float(100 * (1-per.pairwise_global.median()/per.median_knn.median())),
        "pairwise_vs_median_knn_bootstrap95": paired_bootstrap(per, "pairwise_global", "median_knn"),
    }
    if "residual_k20" in per:
        comparisons.update({
            "pairwise_vs_residual_hgb_pct": float(100 * (1-per.pairwise_global.median()/per.residual_k20.median())),
            "pairwise_vs_residual_hgb_bootstrap95": paired_bootstrap(per, "pairwise_global", "residual_k20"),
            "pairwise_better_star_fraction_pct": float(100*(per.pairwise_global < per.residual_k20).mean()),
        })
    payload = {
        "method": "log-flux global calibration with alternating nuisance star intercepts",
        "objective": "within-star centered squared error (equivalent to all-pairs squared differences)",
        "reference_policy": "training stars only; every query TIC excluded",
        "target_derived_star_features": False,
        "rows": {k: int(v.sum()) for k,v in masks.items()},
        "stars": {k: int(df[v].tic_id.nunique()) for k,v in masks.items()},
        "backfit_history": history, "residual_strength": beta,
        "validation_scatter_pct": 100*val_scatter,
        "prior_hgb_alpha": prior_alpha,
        "prior_hgb_validation_scatter_pct": 100*prior_val_scatter,
        "common_test_stars": int(len(per)), "test": summary,
        "comparisons": comparisons, "elapsed_seconds": time.time()-t0,
    }
    (OUT/"results.json").write_text(json.dumps(payload, indent=2))
    per.to_csv(OUT/"per_star_scatter.csv")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
