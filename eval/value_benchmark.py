"""Separate static calibration, temporal references, and learning.

Run from the repository root: python3 eval/value_benchmark.py
Roland's screenshot has no numerical scale: static surfaces here are proxies.
Existing checkpoints are deliberately not used: reference stars and test stars
are disjoint from the fresh model's training stars. No test labels select models.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import cKDTree
from scipy.stats import binned_statistic_2d
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
import torch
import zuko


CONT = ['col', 'row', 'delta_sub_col', 'delta_sub_row', 'sector', 'tmag',
        'crowdsap', 'cdpp1_0', 'pdcvar', 'jitter_rms', 'pdc_noi', 'pr_wght2',
        'gaiarp', 'perstar_gaia_offset', 'sector_ccd_mean_loo', 'spatial_knn_mean_loo']


def cv(frame, offsets):
    work = frame[['tic_id', 'sector_median']].copy()
    work['corrected'] = work.sector_median.to_numpy() / offsets
    g = work.groupby('tic_id').corrected
    return (g.std(ddof=0) / g.mean()).sort_index()


def context(frame):
    return np.column_stack([frame[CONT].to_numpy(dtype=float),
                            np.eye(4)[frame.cam.astype(int).to_numpy()-1],
                            np.eye(4)[frame.ccd.astype(int).to_numpy()-1]])


def static_surface(ref, query, column, sigma):
    """Weighted smoothed 32x32 detector surface, pooled across sectors."""
    result = np.ones(len(query))
    edges = np.linspace(0, 2200, 33)
    centers = (edges[1:] + edges[:-1]) / 2
    for key, q in query.groupby(['cam', 'ccd']):
        r = ref[(ref.cam == key[0]) & (ref.ccd == key[1])]
        if len(r) == 0:
            continue
        counts = np.histogram2d(r.row, r.col, bins=(edges, edges))[0]
        medians = binned_statistic_2d(r.row, r.col, r[column], statistic='median',
                                     bins=(edges, edges)).statistic
        sums = np.nan_to_num(medians)*counts
        mass = gaussian_filter(counts, sigma)
        surface = gaussian_filter(sums, sigma) / np.maximum(mass, 1e-12)
        surface[mass < 1e-6] = r[column].median()
        interpolate = RegularGridInterpolator((centers, centers), surface,
                                               bounds_error=False, fill_value=None)
        xy = np.clip(q[['row', 'col']].to_numpy(), centers[0], centers[-1])
        result[q.index] = interpolate(xy)
    return result


def attach_references(ref, query, k):
    q = query.copy().reset_index(drop=True)
    for name in ['sector_ccd_mean_loo', 'spatial_knn_mean_loo', 'knn_bm', 'knn_gaia',
                 'ccd_median', 'knn_loo_median', 'knn_bm_median', 'knn_gaia_median']:
        q[name] = np.nan
    q['reference_count'] = 0
    groups = {key: r for key, r in ref.groupby(['sector', 'cam', 'ccd'])}
    for key, group in q.groupby(['sector', 'cam', 'ccd']):
        r = groups.get(key)
        if r is None or len(r) < 5:
            continue
        _, idx = cKDTree(r[['col', 'row']].to_numpy()).query(
            group[['col', 'row']].to_numpy(), k=min(k, len(r)))
        idx = np.asarray(idx).reshape(len(group), -1)
        for label, source in [('spatial_knn_mean_loo', 'loo'), ('knn_bm', 'bm'),
                              ('knn_gaia', 'gaia')]:
            values = r[source].to_numpy()[idx]
            q.loc[group.index, label] = values.mean(axis=1)
            q.loc[group.index, 'knn_'+source+'_median'] = np.median(values,axis=1)
        q.loc[group.index, 'sector_ccd_mean_loo'] = r.loo.mean()
        q.loc[group.index, 'ccd_median'] = r.loo.median()
        q.loc[group.index, 'reference_count'] = len(r)
    return q


def flow_train(xtr, ytr, xval, yval, args):
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    ym, ys = float(ytr.mean()), float(ytr.std())
    cfg = dict(features=1, context=xtr.shape[1], transforms=4,
               hidden_features=[64, 64], bins=8)
    if args.full_flow:
        cfg.update(transforms=8, hidden_features=[256, 256], bins=16)
    model = zuko.flows.NSF(**cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-5)
    dataset = torch.utils.data.TensorDataset(torch.tensor(xtr, dtype=torch.float32),
                    torch.tensor((ytr-ym)/ys, dtype=torch.float32).unsqueeze(-1))
    loader = torch.utils.data.DataLoader(dataset, batch_size=512, shuffle=True)
    vx = torch.tensor(xval, dtype=torch.float32)
    vy = torch.tensor((yval-ym)/ys, dtype=torch.float32).unsqueeze(-1)
    best, state, stale = float('inf'), None, 0
    history = []
    if args.reuse_flow:
        saved = torch.load(Path(args.out)/'fresh_flow.pt',map_location='cpu',weights_only=False)
        previous = json.loads((Path(args.out)/'results.json').read_text())
        assert previous['dataset_sha256'] == hashlib.sha256(Path(args.data).read_bytes()).hexdigest()
        for key in ['data','train_stars','val_stars','test_stars','seed','full_flow','epochs']:
            assert saved['args'][key] == vars(args)[key], f'Cannot reuse flow: changed {key}'
        assert saved['flow_info']['config'] == cfg
        assert np.isclose(saved['flow_info']['y_mean'],ym) and np.isclose(saved['flow_info']['y_std'],ys)
        state = saved['model_state']
        history = saved['flow_info']['validation_nll']
        print('Reusing verified fresh checkpoint on the identical split',flush=True)
    for epoch in range(0 if args.reuse_flow else args.epochs):
        model.train()
        for x, y in loader:
            loss = -model(x).log_prob(y).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            losses = [-model(x).log_prob(y).sum().item()
                      for x, y in zip(vx.split(2048), vy.split(2048))]
        nll = sum(losses)/len(vx)
        history.append(nll)
        print(f'Flow epoch {epoch+1}: validation NLL {nll:.4f}', flush=True)
        if nll < best:
            best, state, stale = nll, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
        if stale >= 8:
            break
    model.load_state_dict(state)
    model.eval()

    def predict(x):
        means, stds = [], []
        with torch.no_grad():
            for batch in torch.tensor(x, dtype=torch.float32).split(256):
                s = model(batch).sample((args.samples,)).squeeze(-1)
                means.extend((s.mean(0)*ys+ym).numpy())
                stds.extend((s.std(0)*ys).numpy())
        m, sd = np.asarray(means), np.asarray(stds)
        w = 1/(1+5*sd)
        return 1+w*(m-1)
    return predict, dict(config=cfg, validation_nll=history, y_mean=ym, y_std=ys), state


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', default='training_data_v3.parquet')
    p.add_argument('--out', default='eval/value_benchmark_results')
    p.add_argument('--train-stars', type=int, default=6000)
    p.add_argument('--val-stars', type=int, default=1500)
    p.add_argument('--test-stars', type=int, default=3000)
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--samples', type=int, default=500)
    p.add_argument('--seed', type=int, default=20260915)
    p.add_argument('--bootstrap', type=int, default=1000)
    p.add_argument('--full-flow', action='store_true')
    p.add_argument('--reuse-flow', action='store_true',help='Reuse a completed fresh flow on the identical data/split')
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    df = pd.read_parquet(args.data)
    original = len(df)
    # Select using metadata and measurement validity, never a target-offset cut.
    valid = (df.sector.between(1, 100) & df.cam.isin([1,2,3,4]) &
             df.ccd.isin([1,2,3,4]) & df.tmag.le(13) &
             df.sector_median.gt(0) & df.col.between(0,2200) &
             df.row.between(0,2200) & df.gaiarp.notna())
    df = df[valid].copy()
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['sector_median','gaiarp'])
    if df.duplicated(['tic_id','sector']).any():
        raise ValueError('Duplicate TIC/sector observations')
    counts = df.groupby('tic_id').sector.nunique()
    df = df[df.tic_id.isin(counts[counts >= 5].index)].copy()
    stars = df.groupby('tic_id').cam.agg(lambda x: int(x.mode().iloc[0]))
    refs, rest = train_test_split(stars.index.to_numpy(), test_size=.6,
                                stratify=stars.to_numpy(), random_state=args.seed)
    train, tmp = train_test_split(rest, test_size=1/3, stratify=stars.loc[rest],
                                 random_state=args.seed)
    val, test = train_test_split(tmp, test_size=.5, stratify=stars.loc[tmp],
                                random_state=args.seed)
    def cap(ids, n):
        return rng.choice(ids, min(n,len(ids)), replace=False)
    train, val, test = cap(train,args.train_stars), cap(val,args.val_stars), cap(test,args.test_stars)
    partitions = dict(reference=refs, train=train, validation=val, test=test)
    for a, ia in partitions.items():
        for b, ib in partitions.items():
            if a != b:
                assert not set(ia) & set(ib)
    pd.concat([pd.DataFrame({'tic_id': ids, 'partition':name})
               for name, ids in partitions.items()]).to_csv(out/'split.csv',index=False)
    # Recompute the labels from retained observations, not precomputed parquet labels.
    g = df.groupby('tic_id').sector_median
    total, n = g.transform('sum'), g.transform('count')
    df['loo'] = df.sector_median / ((total-df.sector_median)/(n-1))
    df['bm'] = df.sector_median / (15400*10**((10-df.tmag)/2.5))
    ref = df[df.tic_id.isin(refs)].copy()
    # Gaia-to-TESS calibration fitted ONLY on reference stars, per camera.
    cal = ref.groupby(['tic_id','cam']).agg(flux=('sector_median','median'),rp=('gaiarp','first')).reset_index()
    coefficients = {}
    df['gaia'] = np.nan
    for cam, r in cal.groupby('cam'):
        slope, intercept = np.polyfit(r.rp, np.log10(r.flux), 1)
        coefficients[int(cam)] = [float(slope),float(intercept)]
        mask = df.cam == cam
        df.loc[mask,'gaia'] = df.loc[mask,'sector_median']/10**(slope*df.loc[mask,'gaiarp']+intercept)
    # This observable star-level anchor uses the subject's own flux: keep its
    # role explicit and recompute it in the injection sensitivity experiment.
    df['perstar_gaia_offset'] = df.groupby('tic_id').gaia.transform('mean')
    df['anchor_n'] = df.groupby('tic_id').gaia.transform('count')
    ref = df[df.tic_id.isin(refs)].copy()
    frames = {}
    coverage = {}
    for name, ids in [('train',train),('validation',val),('test',test)]:
        q = attach_references(ref,df[df.tic_id.isin(ids)],20)
        before = len(q)
        unsupported = int(q.reference_count.eq(0).sum())
        q = q.dropna(subset=['spatial_knn_mean_loo']).copy()
        ns = q.groupby('tic_id').sector.nunique()
        q = q[q.tic_id.isin(ns[ns >= 5].index)].reset_index(drop=True)
        frames[name] = q
        coverage[name] = dict(requested_stars=len(ids),evaluated_stars=int(q.tic_id.nunique()),
                              rows=len(q),unsupported_rows=unsupported,
                              rows_removed_for_coverage=before-len(q))
    tr, vl, te = frames['train'], frames['validation'], frames['test']
    print('Partitions:',coverage,flush=True)
    predictions = {'Raw':np.ones(len(te)), 'Sector CCD / LOO':te.sector_ccd_mean_loo.to_numpy(),
                   'Sector KNN / TIC BM':te.knn_bm.to_numpy(), 'Sector KNN / Gaia':te.knn_gaia.to_numpy(),
                   'Sector KNN / LOO':te.spatial_knn_mean_loo.to_numpy(),
                   'Sector CCD median / LOO':te.ccd_median.to_numpy(),
                   'Sector KNN median / TIC BM':te.knn_bm_median.to_numpy(),
                   'Sector KNN median / Gaia':te.knn_gaia_median.to_numpy(),
                   'Sector KNN median / LOO':te.knn_loo_median.to_numpy()}
    selections = {}
    for column, label in [('bm','Static surface / TIC BM (proxy)'),('gaia','Static surface / Gaia')]:
        scores = {}
        for sigma in [1,2,4]:
            scores[sigma] = float(cv(vl,static_surface(ref,vl,column,sigma)).median())
        best = min(scores,key=scores.get)
        selections[label] = dict(sigma=best,validation_cv=scores)
        predictions[label] = static_surface(ref,te,column,best)
    # Test whether a single validation-tuned shrinkage weight closes the flow gap.
    choices = {}
    for column in ['spatial_knn_mean_loo','knn_loo_median']:
        for weight in np.linspace(0,1,21):
            choices[(column,float(weight))] = float(cv(vl,1+weight*(vl[column].to_numpy()-1)).median())
    column, weight = min(choices,key=choices.get)
    simple_label = 'Validation-calibrated LOO KNN'
    selections[simple_label] = dict(column=column,weight=weight,validation_cv=choices[(column,weight)])
    predictions[simple_label] = 1+weight*(te[column].to_numpy()-1)
    prep = make_pipeline(SimpleImputer(strategy='median'),StandardScaler())
    xtr = prep.fit_transform(context(tr)); xv = prep.transform(context(vl)); xt = prep.transform(context(te))
    ytr, yv = tr.loo.to_numpy(), vl.loo.to_numpy()
    learned = {}
    for kind in ['ridge','trees']:
        candidates = ([.1,10,1000] if kind == 'ridge' else [15,31])
        fitted, scores = {}, {}
        for value in candidates:
            model = (Ridge(alpha=value) if kind == 'ridge' else
                     HistGradientBoostingRegressor(max_leaf_nodes=value,max_iter=100,
                                                    l2_regularization=1,random_state=args.seed))
            model.fit(xtr,ytr)
            scores[value] = float(cv(vl,np.clip(model.predict(xv),.5,2)).median())
            fitted[value] = model
        best = min(scores,key=scores.get)
        label = 'Ridge / 24 inputs' if kind == 'ridge' else 'Boosted trees / 24 inputs'
        learned[label] = fitted[best]
        predictions[label] = np.clip(fitted[best].predict(xt),.5,2)
        selections[label] = dict(parameter=best,validation_cv=scores)
    predict_flow, flow_info, flow_state = flow_train(xtr,ytr,xv,yv,args)
    label = 'Fresh full NSF / 24 inputs' if args.full_flow else 'Fresh compact NSF / 24 inputs'
    torch.manual_seed(args.seed+1)
    predictions[label] = np.clip(predict_flow(xt),.5,2)
    torch.save(dict(model_state=flow_state,flow_info=flow_info,continuous_cols=CONT,
                    preprocessing='See preprocessing.npz; continuous then cam/ccd OHE',
                    target='recomputed LOO',args=vars(args)),out/'fresh_flow.pt')
    imputer, scaler = prep.steps[0][1], prep.steps[1][1]
    np.savez(out/'preprocessing.npz',imputation=imputer.statistics_,means=scaler.mean_,stds=scaler.scale_)
    all_cv = pd.DataFrame({name:cv(te,pred) for name,pred in predictions.items()})
    all_cv.to_csv(out/'per_star_cv.csv')
    raw = all_cv.Raw.to_numpy()
    boot = rng.integers(0,len(raw),size=(args.bootstrap,len(raw)))
    rows = []
    for name in all_cv:
        values = all_cv[name].to_numpy()
        reductions = 100*(1-np.median(values[boot],axis=1)/np.median(raw[boot],axis=1))
        rows.append(dict(method=name,median_cv_percent=100*float(np.median(values)),
                         reduction_percent=100*float(1-np.median(values)/np.median(raw)),
                         reduction_ci95=np.quantile(reductions,[.025,.975]).tolist(),
                         stars_worse_percent=100*float(np.mean(values>raw))))
    comparisons = []
    for baseline in ['Static surface / TIC BM (proxy)','Sector KNN / TIC BM',
                     'Sector KNN / Gaia','Sector KNN / LOO','Sector KNN median / TIC BM',
                     'Sector KNN median / LOO',simple_label,'Boosted trees / 24 inputs']:
        b, f = all_cv[baseline].to_numpy(), all_cv[label].to_numpy()
        gain = 100*(1-np.median(f)/np.median(b))
        gains = 100*(1-np.median(f[boot],axis=1)/np.median(b[boot],axis=1))
        comparisons.append(dict(flow_vs=baseline,residual_scatter_gain_percent=float(gain),
                                ci95=np.quantile(gains,[.025,.975]).tolist()))
    # Paired injection response: changes relative to un-injected corrected output.
    # This is sensitivity at sector-median level, not full light-curve validation.
    injections = []
    chosen = rng.choice(all_cv.index,min(200,len(all_cv)),replace=False)
    mask = te.tic_id.isin(chosen).to_numpy()
    sub = te[mask].copy().reset_index(drop=True)
    for period in [54,135,365]:
        phase = dict(zip(chosen,rng.uniform(0,2*np.pi,len(chosen))))
        # Actual sector mid-times are absent: sector number is only a time proxy.
        signal = .02*np.sin(2*np.pi*27.4*sub.sector.to_numpy()/period+sub.tic_id.map(phase).to_numpy())
        inj = sub.copy()
        inj.sector_median *= 1+signal
        inj['anchor_delta'] = sub.gaia.to_numpy()*signal
        inj.perstar_gaia_offset = (sub.perstar_gaia_offset +
            inj.groupby('tic_id').anchor_delta.transform('sum') / sub.anchor_n)
        ix = prep.transform(context(inj))
        for name in ['Sector KNN / LOO',simple_label,*learned,label]:
            base = predictions[name][mask]
            if name in ['Sector KNN / LOO',simple_label]:
                changed = base
            elif name == label:
                # Identical RNG draws isolate input response from sampling noise.
                torch.manual_seed(args.seed)
                base = np.clip(predict_flow(prep.transform(context(sub))),.5,2)
                torch.manual_seed(args.seed)
                changed = np.clip(predict_flow(ix),.5,2)
            else:
                changed = np.clip(learned[name].predict(ix),.5,2)
            response = ((1+signal)*base/changed)-1
            tmp = pd.DataFrame({'tic_id':sub.tic_id,'s':signal,'r':response})
            slopes = tmp.groupby('tic_id')[['s','r']].apply(
                lambda g: np.dot(g.s-g.s.mean(),g.r-g.r.mean())/max(np.sum((g.s-g.s.mean())**2),1e-12))
            injections.append(dict(method=name,period_days=period,amplitude=.02,
                                   median_response_slope=float(slopes.median())))
    report = dict(args=vars(args),dataset_sha256=hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
                  original_rows=original,selected_rows=len(df),coverage=coverage,
                  gaia_calibration=coefficients,selections=selections,flow=flow_info,
                  metrics=rows,paired_comparisons=comparisons,injection_sensitivity=injections,
                  limitations=['Static BM is a reconstruction, not Roland\'s numerical surface.',
                    'Fresh compact flow is not validation of the canonical checkpoint.' if not args.full_flow else
                    'Fresh full architecture; different training recipe from historical checkpoints.',
                    'Source quiet-star sample and Tmag<=13, >=5 sectors selection limit generalization.',
                    'Bootstrap describes test-star uncertainty with models and reference catalog fixed.',
                    'Single training seed; flow means are estimated by Monte Carlo sampling.',
                    'Injection changes the star-level anchor only; quality metrics remain fixed and sector times are approximate.',
                    'Gaia conversion is an empirical per-camera fit, not independent detector ground truth.'])
    (out/'results.json').write_text(json.dumps(report,indent=2))
    lines = ['# STITCH value benchmark','', 'Fresh training; disjoint reference, training, validation, and test stars.', '',
             '| Method | Median CV % | Reduction % (95% CI) | Stars worse % |',
             '|---|---:|---:|---:|']
    for r in rows:
        lo,hi = r['reduction_ci95']
        lines.append(f"| {r['method']} | {r['median_cv_percent']:.3f} | {r['reduction_percent']:.1f} ({lo:.1f}, {hi:.1f}) | {r['stars_worse_percent']:.1f} |")
    lines += ['', 'Flow added value (positive means lower residual scatter):','']
    for r in comparisons:
        lo,hi = r['ci95']
        lines.append(f"- Versus {r['flow_vs']}: {r['residual_scatter_gain_percent']:.1f}% (95% CI {lo:.1f}, {hi:.1f}).")
    lines += ['', 'Limitations:','']+[f'- {s}' for s in report['limitations']]
    lines += ['', 'Sector-median injection sensitivity (ideal response slope = 1):','',
              '| Method | Approximate period days | Median response slope |',
              '|---|---:|---:|']
    for r in injections:
        lines.append(f"| {r['method']} | {r['period_days']} | {r['median_response_slope']:.3f} |")
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10,7))
    names = [r['method'] for r in rows if r['method'] != 'Raw']
    vals = [r['reduction_percent'] for r in rows if r['method'] != 'Raw']
    cis = np.array([r['reduction_ci95'] for r in rows if r['method'] != 'Raw'])
    ax.errorbar(vals,np.arange(len(names)),xerr=[np.maximum(0,np.array(vals)-cis[:,0]),
                np.maximum(0,cis[:,1]-np.array(vals))],fmt='o',capsize=3)
    ax.set_yticks(np.arange(len(names)),names)
    ax.invert_yaxis()
    ax.axvline(0,color='gray',linewidth=1)
    ax.set_xlabel('Reduction in median cross-sector scatter (%) with paired 95% bootstrap intervals')
    architecture = 'full' if args.full_flow else 'compact'
    ax.set_title(f'Fresh benchmark: {len(raw):,} quiet test stars\nStatic BM is a proxy; flow uses the {architecture} architecture')
    fig.tight_layout()
    fig.savefig(out/'comparison.png',dpi=150)
    plt.close(fig)
    print('\n'.join(lines),flush=True)


if __name__ == '__main__':
    main()
