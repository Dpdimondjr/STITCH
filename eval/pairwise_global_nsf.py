"""Conditional NSF on robust within-star centered log-flux residuals.

This is matched to pairwise_global_robust.py: same untruncated population,
TIC split, training-only spatial reference bank, support restriction, and
feature columns.  Star intercepts are nuisance quantities used only to form
training/evaluation labels; TIC identity is never an input or inference-time
quantity.
"""
from pathlib import Path
import copy, json, time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
import zuko

from pairwise_global_calibration import BASE_COLS, FEATURE_COLS, paired_bootstrap
from pairwise_global_robust import group_median, tune_log_residual
from residual_multiscale_experiment import split_stars, spatial_features, scatter_table

SEED = 20260916
OUT = Path("eval/pairwise_global_nsf_results")
REF_LOWER, REF_UPPER = .5, 1.5


def prepare():
    df = pd.read_parquet("training_data_v3.parquet")
    finite = (np.isfinite(df.sector_median) & df.sector_median.gt(0) &
              np.isfinite(df.flux_offset_loo) & df.n_sectors_total.ge(4) &
              df.tmag.le(13) & df.col.notna() & df.row.notna())
    df = df[finite].copy().reset_index(drop=True)
    train_ids, val_ids, test_ids = split_stars(df)
    masks = {"train": df.tic_id.isin(train_ids),
             "val": df.tic_id.isin(val_ids), "test": df.tic_id.isin(test_ids)}
    reference = df[masks["train"]].copy()
    reference["flux_offset_loo"] = reference.flux_offset_loo.clip(REF_LOWER, REF_UPPER)
    frames, feats, X, unsupported = {}, {}, {}, {}
    for name, mask in masks.items():
        frames[name] = df[mask].reset_index(drop=True)
        print("spatial features", name, len(frames[name]), flush=True)
        feats[name] = spatial_features(reference, df[mask])
        X[name] = pd.concat([frames[name][BASE_COLS], feats[name]], axis=1)[FEATURE_COLS]
        good = np.isfinite(feats[name].knn20_median.to_numpy())
        unsupported[name] = int((~good).sum())
        frames[name] = frames[name].loc[good].reset_index(drop=True)
        feats[name] = feats[name].loc[good].reset_index(drop=True)
        X[name] = X[name].loc[good].reset_index(drop=True)
    return frames, feats, X, unsupported


def impute_scale(X):
    med = X["train"].median(numeric_only=True).reindex(FEATURE_COLS).fillna(0.)
    means = X["train"].fillna(med).mean()
    stds = X["train"].fillna(med).std().replace(0, 1).fillna(1)
    arr = {k: ((v.fillna(med)-means)/stds).to_numpy(np.float32) for k,v in X.items()}
    return arr, med, means, stds


def batches_predict(flow, context, device, draws=64, batch=4096):
    medians, qs16, qs84, qs05, qs95 = [], [], [], [], []
    flow.eval()
    with torch.no_grad():
        for start in range(0, len(context), batch):
            c = torch.from_numpy(context[start:start+batch]).to(device)
            s = flow(c).sample((draws,)).squeeze(-1)
            q = torch.quantile(s, torch.tensor([.05,.16,.5,.84,.95],device=device), dim=0)
            qs05.append(q[0].cpu().numpy()); qs16.append(q[1].cpu().numpy())
            medians.append(q[2].cpu().numpy()); qs84.append(q[3].cpu().numpy())
            qs95.append(q[4].cpu().numpy())
    return tuple(np.concatenate(z) for z in (medians,qs16,qs84,qs05,qs95))


def main():
    t0=time.time(); OUT.mkdir(exist_ok=True); np.random.seed(SEED); torch.manual_seed(SEED)
    torch.set_num_threads(4)
    frames, feats, Xdf, unsupported = prepare()
    X, med, means, stds = impute_scale(Xdf)
    base = {k: feats[k].knn20_median.to_numpy() for k in frames}
    response = {k: np.log(frames[k].sector_median.to_numpy())-np.log(base[k]) for k in frames}
    # L1-optimal fixed nuisance intercept. This is evaluation-only for val/test.
    centered = {k: response[k]-group_median(response[k],frames[k].tic_id.to_numpy()) for k in frames}
    ym=float(np.median(centered["train"])); ys=float(np.std(centered["train"]));
    y={k:((v-ym)/ys).astype(np.float32) for k,v in centered.items()}

    config={"features":1,"context":len(FEATURE_COLS),"transforms":4,
            "hidden_features":[128,128],"bins":12}
    flow=zuko.flows.NSF(**config)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); flow.to(device)
    loader=DataLoader(TensorDataset(torch.from_numpy(X["train"]),torch.from_numpy(y["train"]).unsqueeze(1)),
                      batch_size=2048,shuffle=True,generator=torch.Generator().manual_seed(SEED))
    valx=torch.from_numpy(X["val"]).to(device); valy=torch.from_numpy(y["val"]).unsqueeze(1).to(device)
    opt=torch.optim.Adam(flow.parameters(),lr=3e-4,weight_decay=1e-5)
    sched=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,patience=5,factor=.5)
    best=float("inf"); state=None; stale=0; history=[]
    for epoch in range(1,61):
        flow.train(); losses=[]
        for xb,yb in loader:
            xb,yb=xb.to(device),yb.to(device); loss=-flow(xb).log_prob(yb).mean()
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(flow.parameters(),5); opt.step()
            losses.append(loss.item())
        flow.eval()
        with torch.no_grad():
            # Batches avoid retaining the full conditional graph.
            vl=[]
            for s in range(0,len(valx),8192): vl.append(-flow(valx[s:s+8192]).log_prob(valy[s:s+8192]).sum().item())
            vn=sum(vl)/len(valx)
        sched.step(vn); history.append({"epoch":epoch,"train_nll":float(np.mean(losses)),"val_nll":vn})
        print(epoch,history[-1],flush=True)
        if vn < best-1e-4: best=vn; state=copy.deepcopy({k:v.cpu() for k,v in flow.state_dict().items()}); stale=0
        else: stale+=1
        if stale>=12: break
    flow.load_state_dict(state); flow.to(device)
    vpred,*_=batches_predict(flow,X["val"],device,draws=64)
    vpred=vpred*ys+ym
    beta,valscatter=tune_log_residual(frames["val"],np.log(base["val"]),vpred)
    tmed,t16,t84,t05,t95=batches_predict(flow,X["test"],device,draws=128)
    tmed=tmed*ys+ym; t16=t16*ys+ym; t84=t84*ys+ym; t05=t05*ys+ym; t95=t95*ys+ym
    pred=np.exp(np.log(base["test"])+beta*tmed)
    per=scatter_table(frames["test"],{"raw":np.ones(len(pred)),"median_knn":base["test"],"pairwise_global_nsf":pred})
    hgb=pd.read_csv("eval/pairwise_global_robust_results/per_star_scatter.csv",index_col=0)
    common=per.join(hgb[["pairwise_global_robust"]],how="inner")
    # Coverage concerns the centered latent log correction, not absolute flux.
    truth=centered["test"]
    lo68,hi68=np.minimum(beta*t16,beta*t84),np.maximum(beta*t16,beta*t84)
    lo90,hi90=np.minimum(beta*t05,beta*t95),np.maximum(beta*t05,beta*t95)
    coverage={"central_68_pct":float(100*np.mean((truth>=lo68)&(truth<=hi68))),
              "central_90_pct":float(100*np.mean((truth>=lo90)&(truth<=hi90))),
              "median_abs_centered_log_error":float(np.median(np.abs(truth-beta*tmed)))}
    summary={c:{"median_scatter_pct":float(100*per[c].median()),
                "reduction_from_raw_pct":float(100*(1-per[c].median()/per.raw.median())),
                "stars_worse_than_raw_pct":float(100*(per[c]>per.raw).mean())} for c in per}
    comp={"nsf_vs_hgb_pct":float(100*(1-common.pairwise_global_nsf.median()/common.pairwise_global_robust.median())),
          "nsf_vs_hgb_bootstrap95":paired_bootstrap(common,"pairwise_global_nsf","pairwise_global_robust"),
          "nsf_better_star_fraction_pct":float(100*(common.pairwise_global_nsf<common.pairwise_global_robust).mean())}
    payload={"method":"conditional NSF on fixed median-centered within-star log-flux residual",
      "nuisance_intercepts":"Per-star median of log(sector_median)-log(KNN20 median), computed before fitting; val/test medians only score latent coverage and never enter predictions.",
      "limitation":"Fixed centered labels are not alternating joint optimization of nuisance intercepts and the conditional density; the HGB comparator backfits intercepts for five rounds.",
      "selection":"identical to pairwise_global_robust.py; no target-value clipping",
      "reference_policy":"training stars only; own TIC excluded; reference labels bounded [0.5,1.5]",
      "features":FEATURE_COLS,"imputation":"training medians; training means/stds", "architecture":config,
      "device":str(device),"epochs":len(history),"best_val_nll":best,"beta":beta,
      "validation_scatter_pct":100*valscatter,"rows":{k:len(v) for k,v in frames.items()},
      "stars":{k:int(v.tic_id.nunique()) for k,v in frames.items()},"unsupported_rows":unsupported,
      "test":summary,"comparison":comp,"predictive_interval_coverage":coverage,"elapsed_seconds":time.time()-t0}
    (OUT/"results.json").write_text(json.dumps(payload,indent=2)); per.to_csv(OUT/"per_star_scatter.csv")
    torch.save({"state":state,"config":config,"feature_cols":FEATURE_COLS,"medians":med,
                "means":means,"stds":stds,"y_median":ym,"y_std":ys},OUT/"model.pt")
    print(json.dumps(payload,indent=2))

if __name__=="__main__": main()
