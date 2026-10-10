"""Compare an existing checkpoint to robust KNN on its reconstructed split.

Verify all saved training normalization statistics before describing test TICs
as held out. Report historical all-data LOO context and train-only reference
context separately; the latter changes deployment context without retraining.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import zuko
from scipy.spatial import cKDTree
from sklearn.model_selection import train_test_split
from value_benchmark import cv, attach_references, static_surface


def historical_features(df, target):
    df = df.reset_index(drop=True).copy()
    groups = df.groupby(['sector','cam','ccd'])[target]
    df['sector_ccd_mean_loo'] = (groups.transform('sum')-df[target])/(groups.transform('count')-1).clip(lower=1)
    means = np.full(len(df),np.nan,dtype=np.float32)
    medians = np.full(len(df),np.nan,dtype=np.float32)
    corrected_means = np.full(len(df),np.nan,dtype=np.float32)
    self_errors = 0
    for _, g in df.groupby(['sector','cam','ccd']):
        if len(g)<2:
            continue
        _, nn = cKDTree(g[['col','row']].to_numpy(dtype=np.float32)).query(
            g[['col','row']].to_numpy(dtype=np.float32),k=min(20,len(g)-1)+1)
        # Reproduce the trainer exactly, and audit its assumption that self is first.
        self_errors += int(np.sum(nn[:,0] != np.arange(len(g))))
        values = g[target].to_numpy()[nn[:,1:]]
        means[g.index] = values.mean(axis=1)
        corrected_nn = np.asarray([neighbors[neighbors!=i][:nn.shape[1]-1]
                                    for i,neighbors in enumerate(nn)])
        corrected_values = g[target].to_numpy()[corrected_nn]
        corrected_means[g.index] = corrected_values.mean(axis=1)
        medians[g.index] = np.median(corrected_values,axis=1)
    df['spatial_knn_mean_loo'] = pd.Series(means).fillna(df.sector_ccd_mean_loo)
    df['knn_loo_median'] = pd.Series(medians).fillna(df.sector_ccd_mean_loo)
    df['knn_loo_corrected_mean'] = pd.Series(corrected_means).fillna(df.sector_ccd_mean_loo)
    df['_record_id'] = np.arange(len(df))
    return df,self_errors


def make_context(frame,ck):
    cont = (frame[ck['continuous_cols']]-ck['means'])/ck['stds']
    cam = pd.get_dummies(frame.cam.astype(int),prefix='cam').reindex(columns=ck['cam_cols'],fill_value=0)
    ccd = pd.get_dummies(frame.ccd.astype(int),prefix='ccd').reindex(columns=ck['ccd_cols'],fill_value=0)
    values = np.column_stack([cont,cam,ccd]).astype(np.float32)
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite checkpoint context')
    return values


def predict(frame,ck,flow,samples,seed):
    torch.manual_seed(seed)
    results = []
    with torch.no_grad():
        for batch in torch.tensor(make_context(frame,ck)).split(256):
            s = flow(batch).sample((samples,)).squeeze(-1)
            mean = s.mean(0).numpy()*ck['y_std']+ck['y_mean']
            sd = s.std(0).numpy()*ck['y_std']
            results.extend(1+(mean-1)/(1+5*sd))
    return np.asarray(results)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',default='training_data_topup_pdc.parquet')
    p.add_argument('--model',default='stitch_nsf_knn.pt')
    p.add_argument('--out',default='eval/checkpoint_value_results')
    p.add_argument('--target',default='flux_offset_loo',choices=['flux_offset_loo'],
                   help='This benchmark currently supports LOO-trained checkpoints only')
    p.add_argument('--samples',type=int,default=500)
    p.add_argument('--seed',type=int,default=20260915)
    args = p.parse_args()
    torch.set_num_threads(4)
    out = Path(args.out); out.mkdir(parents=True,exist_ok=True)
    data_hash = hashlib.sha256(Path(args.data).read_bytes()).hexdigest()
    ck = torch.load(args.model,map_location='cpu',weights_only=False)
    df = pd.read_parquet(args.data)
    # The original trainer constructs the star anchor before its target cut.
    df['perstar_gaia_offset'] = df.groupby('tic_id').flux_offset.transform('mean')
    df = df.dropna(subset=['col','row',args.target])
    df = df[(df[args.target]>.85)&(df[args.target]<1.15)&df.n_sectors_total.ge(4)&df.tmag.le(13)].copy()
    spatial_cols = {'sector_ccd_mean_loo','spatial_knn_mean_loo',
                    'spatial_knn_median_loo','spatial_knn_mad_loo'}
    for col in ck['continuous_cols']:
        if col not in spatial_cols:
            df[col] = df[col].fillna(df[col].median())
    # Robust-mode trainer retains the parquet's precomputed mean-KNN feature;
    # legacy mode recomputes it. Preserve that exact distinction for audit.
    precomputed_mean = None
    if 'spatial_knn_median_loo' in ck['continuous_cols']:
        source = ('spatial_knn_mean_loo' if 'spatial_knn_mean_loo' in df.columns
                  else 'spatial_knn_median_loo')
        precomputed_mean = df[source].reset_index(drop=True).copy()
    df,self_errors = historical_features(df,args.target)
    if precomputed_mean is not None:
        df['spatial_knn_mean_loo'] = precomputed_mean
    if 'spatial_knn_median_loo' in ck['continuous_cols']:
        df['spatial_knn_median_loo'] = df['spatial_knn_median_loo'].fillna(df.sector_ccd_mean_loo)
    if 'spatial_knn_mad_loo' in ck['continuous_cols']:
        df['spatial_knn_mad_loo'] = df['spatial_knn_mad_loo'].fillna(0.0)
    stars = df.groupby('tic_id').cam.agg(lambda x:x.mode().iloc[0]).reset_index()
    train,tmp = train_test_split(stars.tic_id,test_size=.2,stratify=stars.cam,random_state=42)
    # Intentionally reproduce the historical trainer's second split ordering.
    val,test = train_test_split(tmp,test_size=.5,
        stratify=stars[stars.tic_id.isin(tmp)].cam.to_numpy(),random_state=42)
    training = df[df.tic_id.isin(train)]
    cols = ck['continuous_cols']
    observed_means = training[cols].mean()
    observed_stds = training[cols].std().replace(0,1)
    mean_match = np.allclose(observed_means,ck['means'][cols],rtol=1e-8,atol=1e-10)
    std_match = np.allclose(observed_stds,ck['stds'][cols],rtol=1e-8,atol=1e-10)
    target_match = bool(np.allclose([training[args.target].mean(),training[args.target].std()],
                                   [ck['y_mean'],ck['y_std']],rtol=1e-8,atol=1e-10))
    audit = dict(training_stars=int(training.tic_id.nunique()),training_rows=len(training),
                 means_match=bool(mean_match),stds_match=bool(std_match),target_match=target_match,
                 maximum_mean_difference=float(np.max(np.abs(observed_means-ck['means'][cols]))),
                 maximum_std_difference=float(np.max(np.abs(observed_stds-ck['stds'][cols]))),
                 self_first_assumption_errors=self_errors,
                 split_order='Exact reproduction of historical trainer, including second-split strata ordering')
    print('Checkpoint provenance audit:',audit,flush=True)
    (out/'provenance.json').write_text(json.dumps(audit,indent=2))
    if not (mean_match and std_match and target_match):
        raise ValueError('Checkpoint preprocessing does not match this dataset/split; refusing held-out claim')
    pd.concat([pd.DataFrame({'tic_id':ids,'partition':label}) for label,ids in
               [('train',train),('validation',val),('test',test)]]).to_csv(out/'split.csv',index=False)
    df['loo'] = df[args.target]
    df['bm'] = df.sector_median/(15400*10**((10-df.tmag)/2.5))
    df['gaia'] = df.flux_offset
    reference = df[df.tic_id.isin(train)].copy()
    frames = {}
    coverage = {}
    for label,ids in [('validation',val),('test',test)]:
        subjects = df[df.tic_id.isin(ids)&df.sector_median.gt(0)].copy()
        limited = attach_references(reference,subjects,20)
        unsupported = int(limited.reference_count.eq(0).sum())
        limited = limited.dropna(subset=['spatial_knn_mean_loo'])
        counts = limited.groupby('tic_id').sector.nunique()
        limited = limited[limited.tic_id.isin(counts[counts>=5].index)].reset_index(drop=True)
        limited['knn_loo_corrected_mean'] = limited.spatial_knn_mean_loo
        original = df.loc[limited._record_id].reset_index(drop=True)
        assert original.tic_id.equals(limited.tic_id) and original.sector.equals(limited.sector)
        frames[label] = dict(historical=original,train_only=limited)
        coverage[label] = dict(evaluated_stars=int(limited.tic_id.nunique()),rows=len(limited),
                               unsupported_rows=unsupported)
    test_frame = frames['test']['historical']
    print('Common evaluation coverage:',coverage,flush=True)
    flow = zuko.flows.NSF(**ck['flow_config']); flow.load_state_dict(ck['model_state']); flow.eval()
    predictions = {'Raw':np.ones(len(test_frame))}
    choices = {}
    for regime in ['historical','train_only']:
        te,vl = frames['test'][regime],frames['validation'][regime]
        names = [('Mean LOO KNN','knn_loo_corrected_mean'),('Median LOO KNN','knn_loo_median')]
        for name,column in names:
            predictions[f'{regime}: {name}'] = te[column].to_numpy()
        scores = {(column,float(w)):float(cv(vl,1+w*(vl[column].to_numpy()-1)).median())
                  for _,column in names for w in np.linspace(0,1,21)}
        column,w = min(scores,key=scores.get)
        choices[regime] = dict(column=column,weight=w,validation_cv=scores[(column,w)])
        predictions[f'{regime}: Tuned LOO KNN'] = 1+w*(te[column].to_numpy()-1)
        print(f'Predicting checkpoint with {regime} reference context',flush=True)
        predictions[f'{regime}: Existing NSF checkpoint'] = predict(te,ck,flow,args.samples,args.seed)
    vl = frames['validation']['historical']
    scores = {s:float(cv(vl,static_surface(reference,vl,'bm',s)).median()) for s in [1,2,4]}
    sigma = min(scores,key=scores.get)
    choices['static_BM'] = dict(sigma=sigma,validation_cv=scores)
    predictions['Static TIC BM surface (proxy; train references)'] = static_surface(reference,test_frame,'bm',sigma)
    if hashlib.sha256(Path(args.data).read_bytes()).hexdigest() != data_hash:
        raise ValueError('Source data changed during evaluation')
    cvs = pd.DataFrame({label:cv(test_frame,offset) for label,offset in predictions.items()})
    cvs.to_csv(out/'per_star_cv.csv')
    observations = test_frame[['tic_id','sector','cam','ccd','sector_median']].copy()
    for label,offset in predictions.items():
        observations[label] = offset
    observations.to_parquet(out/'per_observation_offsets.parquet',index=False)
    rng = np.random.default_rng(args.seed)
    boot = rng.integers(0,len(cvs),size=(1000,len(cvs)))
    raw = cvs.Raw.to_numpy()
    metrics = []
    for label in cvs:
        x = cvs[label].to_numpy()
        reduction = 100*(1-np.median(x[boot],axis=1)/np.median(raw[boot],axis=1))
        metrics.append(dict(method=label,median_cv_percent=100*float(np.median(x)),
                            reduction_percent=100*float(1-np.median(x)/np.median(raw)),
                            reduction_ci95=np.quantile(reduction,[.025,.975]).tolist(),
                            stars_worse_percent=100*float(np.mean(x>raw))))
    comparisons = []
    for regime in ['historical','train_only']:
        f = cvs[f'{regime}: Existing NSF checkpoint'].to_numpy()
        for baseline in ['Mean LOO KNN','Median LOO KNN','Tuned LOO KNN']:
            b = cvs[f'{regime}: {baseline}'].to_numpy()
            gain = 100*(1-np.median(f[boot],axis=1)/np.median(b[boot],axis=1))
            comparisons.append(dict(regime=regime,baseline=baseline,
                flow_residual_scatter_gain_percent=100*float(1-np.median(f)/np.median(b)),
                ci95=np.quantile(gain,[.025,.975]).tolist()))
    limitations = ["Static TIC BM surface is a proxy, not Roland's measured correction.",
        'Historical regime uses LOO neighbor labels from all partitions, as during training; test labels influence other subjects\' inputs.',
        f'Historical trainer assumed self was the first nearest neighbor; that assumption fails in {self_errors} tied-position records. Checkpoint context reproduces the trainer; direct KNN baselines explicitly exclude self.',
        'Train-only regime excludes test/validation stars from spatial references, but checkpoint training used historical all-partition references; this is not a clean independent retraining.',
        'Original target-offset filtering is reproduced for checkpoint compatibility; evaluation does not represent all stars.',
        'Saved normalization matches are strong reconstruction evidence, not a substitute for saved training TIC IDs.',
        'Bootstrap holds the checkpoint/reference catalog fixed; no real-variable signal-preservation claim.',
        'Per-star Gaia anchor uses the subject\'s own flux and the historical empirical Gaia calibration.']
    report = dict(args=vars(args),model_sha256=hashlib.sha256(Path(args.model).read_bytes()).hexdigest(),
                  data_sha256=data_hash,audit=audit,coverage=coverage,choices=choices,
                  metrics=metrics,comparisons=comparisons,limitations=limitations)
    (out/'results.json').write_text(json.dumps(report,indent=2))
    lines = ['# Existing-checkpoint comparison','',f"Checkpoint: `{args.model}`; data: `{args.data}`.",
        f"Saved means, standard deviations and target scaling match the reconstructed {audit['training_stars']:,}-star training partition.",
        f"All methods evaluated on the same {len(cvs):,} test stars with at least five supported sectors.",'',
        '| Method | Median scatter % | Reduction % (95% CI) | Stars worse % |','|---|---:|---:|---:|']
    for r in metrics:
        lo,hi = r['reduction_ci95']
        lines.append(f"| {r['method']} | {r['median_cv_percent']:.3f} | {r['reduction_percent']:.1f} ({lo:.1f}, {hi:.1f}) | {r['stars_worse_percent']:.1f} |")
    lines += ['', 'Checkpoint gain in residual scatter (positive means better):','']
    for r in comparisons:
        lo,hi = r['ci95']
        lines.append(f"- {r['regime']} vs {r['baseline']}: {r['flow_residual_scatter_gain_percent']:.1f}% (95% CI {lo:.1f}, {hi:.1f}).")
    lines += ['', 'Limitations:','']+[f'- {x}' for x in limitations]
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines),flush=True)


if __name__ == '__main__':
    main()
