"""Temporal deployment test for leakage-safe STITCH-style calibration.

Model fitting uses sectors <=59.  Validation uses sectors 60--67 and the
reported future test uses sectors 68--85.  TICs are permanently assigned to
one of three roles: calibrator/model-training, validation science target, or
test science target.  Thus no evaluated target supplies a label or feature.

At deployment, calibrator stars observed in each new sector supply the local
KNN reference labels.  This deliberately tests the realistic regime in which
a calibration catalogue is available in a new sector, while science targets
remain completely star-disjoint from it.
"""
from pathlib import Path
import json, time
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split

from residual_multiscale_experiment import spatial_features, scatter_table
from pairwise_global_robust import fit_robust_global, tune_log_residual
from pairwise_global_calibration import BASE_COLS, FEATURE_COLS, paired_bootstrap

SEED = 20260916
OUT = Path("eval/future_sector_generalization_results")
TRAIN_MAX, VAL_MIN, VAL_MAX, TEST_MIN, TEST_MAX = 59, 60, 67, 68, 85


def assign_roles(df):
    """Permanent star-disjoint roles, stratified by dominant camera."""
    stars = (df.groupby("tic_id").cam.agg(lambda x: x.mode().iloc[0])
             .rename("cam").reset_index())
    calibrator, science = train_test_split(
        stars, test_size=.30, stratify=stars.cam, random_state=SEED)
    validation, test = train_test_split(
        science, test_size=.50, stratify=science.cam, random_state=SEED)
    return {"calibrator": set(calibrator.tic_id),
            "validation": set(validation.tic_id), "test": set(test.tic_id)}


def supported_frame(query, features, cols):
    X = pd.concat([query.reset_index(drop=True), features], axis=1)[cols]
    good = np.isfinite(features.knn20_median.to_numpy())
    return (query.reset_index(drop=True).loc[good].reset_index(drop=True),
            features.loc[good].reset_index(drop=True),
            X.loc[good].reset_index(drop=True))


def tune_linear(frame, base, residual, log=False):
    best = None
    for strength in np.linspace(-.25, 1.5, 36):
        pred = (np.exp(np.log(base) + strength*residual) if log
                else base + strength*residual)
        score = float(scatter_table(frame, {"x": pred}).x.median())
        if best is None or score < best[1]: best=(float(strength),score)
    return best


def coverage_table(ref, query, feats):
    rows=[]
    for sector, q in query.groupby("sector"):
        pos=q.index.to_numpy()
        # query is reset before spatial_features, so these positions align.
        ok=np.isfinite(feats.iloc[pos].knn20_median.to_numpy())
        r=ref[ref.sector.eq(sector)]
        rows.append({"sector":int(sector), "calibrator_rows":len(r),
          "calibrator_stars":int(r.tic_id.nunique()), "target_rows":len(q),
          "target_stars":int(q.tic_id.nunique()), "supported_rows":int(ok.sum()),
          "support_fraction":float(ok.mean())})
    return pd.DataFrame(rows)


def main():
    t0=time.time(); OUT.mkdir(exist_ok=True)
    df=pd.read_parquet("training_data_v3.parquet")
    finite=(np.isfinite(df.sector_median)&df.sector_median.gt(0)&
            np.isfinite(df.flux_offset_loo)&df.n_sectors_total.ge(4)&
            df.tmag.le(13)&df.col.notna()&df.row.notna()&
            df.sector.between(1,TEST_MAX))
    df=df[finite].copy().reset_index(drop=True)
    roles=assign_roles(df)
    calib=df[df.tic_id.isin(roles["calibrator"])].copy()
    periods={
      "train":(calib[calib.sector.le(TRAIN_MAX)].copy(),
               calib[calib.sector.le(TRAIN_MAX)].copy()),
      "validation":(calib[calib.sector.between(VAL_MIN,VAL_MAX)].copy(),
                    df[df.tic_id.isin(roles["validation"]) &
                       df.sector.between(VAL_MIN,VAL_MAX)].copy()),
      "test":(calib[calib.sector.between(TEST_MIN,TEST_MAX)].copy(),
              df[df.tic_id.isin(roles["test"]) &
                 df.sector.between(TEST_MIN,TEST_MAX)].copy()),
    }
    frames={}; feats={}; Xfull={}; coverage={}
    for name,(ref,query) in periods.items():
        # Bound only calibrator labels used to construct a robust local prior.
        ref=ref.copy(); ref["flux_offset_loo"]=ref.flux_offset_loo.clip(.5,1.5)
        query=query.reset_index(drop=True)
        print("features",name,"refs",len(ref),"queries",len(query),flush=True)
        f=spatial_features(ref,query)
        coverage[name]=coverage_table(ref,query,f)
        frames[name],feats[name],Xfull[name]=supported_frame(query,f,FEATURE_COLS)
    # No-sector feature set measures whether a raw numeric sector covariate
    # helps when every future value lies beyond the training maximum.
    no_sector=[c for c in FEATURE_COLS if c != "sector"]

    base={k:feats[k].knn20_median.to_numpy() for k in frames}
    # Direct robust residual learner over the local KNN calibration.
    hgb=HistGradientBoostingRegressor(loss="absolute_error",learning_rate=.06,
        max_iter=250,max_leaf_nodes=31,min_samples_leaf=40,l2_regularization=.1,
        random_state=SEED,early_stopping=False)
    hgb.fit(Xfull["train"],frames["train"].flux_offset_loo.to_numpy()-base["train"])
    alpha,_=tune_linear(frames["validation"],base["validation"],
                         hgb.predict(Xfull["validation"]))
    residual_pred=base["test"]+alpha*hgb.predict(Xfull["test"])

    global_predictions={}; global_meta={}
    for label,cols in [("robust_global",FEATURE_COLS),
                       ("robust_global_no_sector",no_sector)]:
        model,history=fit_robust_global(
          Xfull["train"][cols],np.log(frames["train"].sector_median.to_numpy()),
          np.log(base["train"]),frames["train"].tic_id.to_numpy())
        beta,valscore=tune_log_residual(
          frames["validation"],np.log(base["validation"]),
          model.predict(Xfull["validation"][cols]))
        global_predictions[label]=np.exp(np.log(base["test"])+
          beta*model.predict(Xfull["test"][cols]))
        global_meta[label]={"beta":beta,"validation_scatter_pct":100*valscore,
                            "history":history}

    preds={"raw":np.ones(len(frames["test"])),"median_knn":base["test"],
           "residual_hgb":residual_pred,**global_predictions}
    per=scatter_table(frames["test"],preds)
    summary={c:{"median_scatter_pct":float(100*per[c].median()),
      "reduction_from_raw_pct":float(100*(1-per[c].median()/per.raw.median())),
      "stars_worse_than_raw_pct":float(100*(per[c]>per.raw).mean())} for c in per}
    comparisons={}
    for method in ["residual_hgb","robust_global","robust_global_no_sector"]:
      comparisons[method+"_vs_median_knn_pct"]=float(100*(1-per[method].median()/per.median_knn.median()))
      comparisons[method+"_vs_median_knn_bootstrap95"]=paired_bootstrap(per,method,"median_knn",draws=2000)
    comparisons["raw_sector_increment_pct"]=float(100*(1-per.robust_global.median()/per.robust_global_no_sector.median()))
    comparisons["raw_sector_increment_bootstrap95"]=paired_bootstrap(per,"robust_global","robust_global_no_sector",draws=2000)
    cov={k:{"reference_rows":int(len(periods[k][0])),
            "query_rows":int(len(periods[k][1])),
            "supported_rows":int(len(frames[k])),
            "support_fraction":float(len(frames[k])/max(1,len(periods[k][1]))),
            "sectors":sorted(int(x) for x in periods[k][1].sector.unique())}
         for k in periods}
    payload={"temporal_split":{"model_fit_sectors":"1-59",
      "validation_sectors":"60-67","future_test_sectors":"68-85"},
      "role_policy":"70% calibrator, 15% validation science, 15% test science; TIC-disjoint permanently",
      "deployment_policy":"new-sector KNN uses labels from calibrator-role stars in that sector; evaluated target stars never supply labels",
      "selection":"finite positive sector_median; finite LOO; N>=4; Tmag<=13; valid position; no target-value clipping",
      "prediction_bound":"scatter evaluator clips applied offsets to [0.85,1.15] for every learned method",
      "target_derived_subject_features":False,"coverage":cov,
      "test_stars_with_at_least_5_supported_visits":int(len(per)),
      "residual_hgb_alpha":alpha,"global_models":global_meta,"test":summary,
      "comparisons":comparisons,"elapsed_seconds":time.time()-t0}
    (OUT/"results.json").write_text(json.dumps(payload,indent=2))
    per.to_csv(OUT/"per_star_scatter.csv")
    pd.concat([v.assign(period=k) for k,v in coverage.items()]).to_csv(OUT/"sector_coverage.csv",index=False)
    print(json.dumps(payload,indent=2))

if __name__=="__main__": main()
