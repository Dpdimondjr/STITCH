"""Robust, untruncated global log-flux calibration experiment.

No observation is selected by its flux-offset target. Reference labels alone
are bounded before constructing median-KNN features. The shared model and
nuisance star intercepts minimize an alternating L1 objective.
"""
from pathlib import Path
import json, time
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from pairwise_global_calibration import (BASE_COLS, FEATURE_COLS,
    paired_bootstrap)
from residual_multiscale_experiment import (split_stars, spatial_features,
    scatter_table, tune_alpha)

SEED = 20260916
OUT = Path("eval/pairwise_global_robust_results")
REF_LOWER, REF_UPPER = 0.5, 1.5


def group_median(values, ids):
    return (pd.Series(values).groupby(pd.Series(ids))
            .transform("median").to_numpy())


def robust_center(values, ids):
    return values - group_median(values, ids)


def fit_robust_global(X, log_flux, log_base, ids, rounds=5):
    response = log_flux - log_base
    intercept = group_median(response, ids)
    history = []
    model = None
    for iteration in range(rounds):
        model = HistGradientBoostingRegressor(
            loss="absolute_error", learning_rate=.06, max_iter=250,
            max_leaf_nodes=31, min_samples_leaf=40, l2_regularization=.1,
            random_state=SEED + iteration, early_stopping=False)
        model.fit(X, response - intercept)
        fitted = model.predict(X)
        intercept = group_median(response - fitted, ids)
        centered = robust_center(response - fitted, ids)
        history.append({
            "iteration": iteration + 1,
            "within_star_median_absolute_log_error": float(np.median(np.abs(centered))),
            "within_star_mean_absolute_log_error": float(np.mean(np.abs(centered))),
        })
    return model, history


def tune_log_residual(frame, log_base, residual):
    best = None
    for beta in np.linspace(-.25, 1.5, 36):
        pred = np.exp(log_base + beta*residual)
        score = float(scatter_table(frame, {"candidate": pred}).candidate.median())
        if best is None or score < best[1]:
            best = (float(beta), score)
    return best


def main():
    t0=time.time(); OUT.mkdir(exist_ok=True)
    df=pd.read_parquet("training_data_v3.parquet")
    # flux_offset_loo is required only for the matched KNN and prior residual
    # baseline. It is never used to select by value.
    finite = (np.isfinite(df.sector_median) & df.sector_median.gt(0) &
              np.isfinite(df.flux_offset_loo) & df.n_sectors_total.ge(4) &
              df.tmag.le(13) & df.col.notna() & df.row.notna())
    df=df[finite].copy().reset_index(drop=True)
    train_ids,val_ids,test_ids=split_stars(df)
    masks={"train":df.tic_id.isin(train_ids),"val":df.tic_id.isin(val_ids),
           "test":df.tic_id.isin(test_ids)}
    reference=df[masks["train"]].copy()
    raw_ref=reference.flux_offset_loo.to_numpy(copy=True)
    reference["flux_offset_loo"]=reference.flux_offset_loo.clip(REF_LOWER,REF_UPPER)
    bounded_fraction=float(np.mean((raw_ref<REF_LOWER)|(raw_ref>REF_UPPER)))
    frames={}; feats={}; X={}; unsupported={}
    for name,mask in masks.items():
        frames[name]=df[mask].reset_index(drop=True)
        print("spatial features",name,len(frames[name]),flush=True)
        feats[name]=spatial_features(reference,df[mask])
        X[name]=pd.concat([frames[name][BASE_COLS],feats[name]],axis=1)[FEATURE_COLS]
        good=np.isfinite(feats[name].knn20_median.to_numpy())
        unsupported[name]=int((~good).sum())
        frames[name]=frames[name].loc[good].reset_index(drop=True)
        feats[name]=feats[name].loc[good].reset_index(drop=True)
        X[name]=X[name].loc[good].reset_index(drop=True)

    base={k:feats[k].knn20_median.to_numpy() for k in frames}
    model,history=fit_robust_global(
        X["train"],np.log(frames["train"].sector_median.to_numpy()),
        np.log(base["train"]),frames["train"].tic_id.to_numpy())
    rv=model.predict(X["val"])
    beta,val_scatter=tune_log_residual(frames["val"],np.log(base["val"]),rv)
    pair_pred=np.exp(np.log(base["test"])+beta*model.predict(X["test"]))

    # Matched leakage-safe residual baseline, trained without target clipping.
    prior=HistGradientBoostingRegressor(
        loss="absolute_error",learning_rate=.06,max_iter=250,max_leaf_nodes=31,
        min_samples_leaf=40,l2_regularization=.1,random_state=20260915,
        early_stopping=False)
    prior.fit(X["train"],frames["train"].flux_offset_loo.to_numpy()-base["train"])
    pv=base["val"]+prior.predict(X["val"])
    alpha,prior_val=tune_alpha(frames["val"],pv)
    prior_pred=1+alpha*(base["test"]+prior.predict(X["test"])-1)

    # Two MAE diagnostics, kept distinct from per-star scatter: prediction error
    # against the noisy LOO label, and centered log-flux error after correction.
    target_test=frames["test"].flux_offset_loo.to_numpy()
    def centered_log_mae(offset):
        e=np.log(frames["test"].sector_median.to_numpy())-np.log(offset)
        return float(np.mean(np.abs(e-pd.Series(e).groupby(frames["test"].tic_id).transform('median').to_numpy())))
    mae_diag={
      'loo_label_mae': {k:float(np.mean(np.abs(v-target_test))) for k,v in {
          'median_knn':base['test'],'residual_hgb':prior_pred,
          'robust_global':pair_pred}.items()},
      'centered_log_flux_mae': {k:centered_log_mae(v) for k,v in {
          'raw':np.ones(len(frames['test'])),'median_knn':base['test'],
          'residual_hgb':prior_pred,'robust_global':pair_pred}.items()}}

    per=scatter_table(frames["test"],{"raw":np.ones(len(frames["test"])),
        "median_knn":base["test"],"residual_k20":prior_pred,
        "pairwise_global_robust":pair_pred})
    summary={c:{"median_scatter_pct":float(100*per[c].median()),
        "reduction_from_raw_pct":float(100*(1-per[c].median()/per.raw.median())),
        "stars_worse_than_raw_pct":float(100*(per[c]>per.raw).mean())} for c in per}
    comparisons={
        "pairwise_vs_median_knn_pct":float(100*(1-per.pairwise_global_robust.median()/per.median_knn.median())),
        "pairwise_vs_median_knn_bootstrap95":paired_bootstrap(per,"pairwise_global_robust","median_knn"),
        "pairwise_vs_residual_hgb_pct":float(100*(1-per.pairwise_global_robust.median()/per.residual_k20.median())),
        "pairwise_vs_residual_hgb_bootstrap95":paired_bootstrap(per,"pairwise_global_robust","residual_k20"),
        "pairwise_better_star_fraction_pct":float(100*(per.pairwise_global_robust<per.residual_k20).mean()),
    }
    payload={"method":"robust global log-flux calibration with median star intercepts",
      "selection":"positive finite sector_median and flux_offset_loo; n_sectors>=4; tmag<=13; no target-value clipping",
      "reference_policy":"training stars only; own TIC excluded; reference flux_offset_loo clipped to [0.5,1.5]",
      "reference_labels_bounded_fraction_pct":100*bounded_fraction,
      "unsupported_rows_removed":unsupported,
      "rows_after_support":{k:len(v) for k,v in frames.items()},
      "stars_after_support":{k:int(v.tic_id.nunique()) for k,v in frames.items()},
      "history":history,"residual_strength":beta,
      "validation_scatter_pct":100*val_scatter,"prior_hgb_alpha":alpha,
      "prior_hgb_validation_scatter_pct":100*prior_val,
      "common_test_stars":len(per),"test":summary,"mae_diagnostics":mae_diag,"comparisons":comparisons,
      "elapsed_seconds":time.time()-t0}
    (OUT/"results.json").write_text(json.dumps(payload,indent=2))
    per.to_csv(OUT/"per_star_scatter.csv")
    print(json.dumps(payload,indent=2))


if __name__=="__main__": main()
