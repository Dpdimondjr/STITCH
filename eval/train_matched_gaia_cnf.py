"""Train a CNF on the exact frozen physical HGB rows, inputs and Gaia target.

No acquisition; existing research inputs are read-only. Calibration alone selects
the checkpoint. Confirmation/development-test predictions are made after freezing.
"""
import os
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import argparse
import copy
import gc
import hashlib
import json
import shutil
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import zuko

BASE = Path('eval/full_rebuild_20261012')
ROOT = Path('eval/matched_gaia_cnf_20261017')
LABELS = Path('eval/hgb_reference_repair_20261010/clean_features.parquet')
SEED = 2026101201
CONFIG = dict(features=1, context=120, transforms=4, hidden_features=[128, 128], bins=12)
MAX_EPOCHS = 60
PATIENCE = 12
BATCH = 2048
META = ['source_row', 'tic_id', 'sector', 'role', 'physical_group_id',
        'reference_fold', 'evaluation_group', 'fit_row', 'systematic_score',
        'has_gaia_anchor', 'gaia_log_anchor', 'pred_knn']


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def checkpoint(path, obj):
    tmp = path.with_suffix('.tmp')
    torch.save(obj, tmp)
    tmp.replace(path)


def reserve():
    free = shutil.disk_usage('.').free / 2**30
    if free < 4:
        raise RuntimeError(f'Preserving 4 GiB disk reserve; currently {free:.2f} GiB free')
    return free


def names():
    return read(BASE/'physical/complete.json')['features']


def labels(frame):
    y = pd.read_parquet(LABELS, columns=['source_row', 'sector_median'])
    assert not y.source_row.duplicated().any()
    values = y.set_index('source_row').sector_median.reindex(frame.source_row).to_numpy()
    assert np.isfinite(values).all() and (values > 0).all()
    return values


def target(frame, flux):
    value = np.log(flux/frame.pred_knn.to_numpy())-frame.gaia_log_anchor.to_numpy()
    missing = ~frame.has_gaia_anchor.to_numpy()
    if missing.any():
        v = pd.Series(value[missing])
        value[missing] -= v.groupby(pd.Series(frame.tic_id.to_numpy()[missing])).transform('median').to_numpy()
    return value


def cv_values(frame, correction):
    values = pd.DataFrame({'tic_id':frame.tic_id.to_numpy(),
                           'sector':frame.sector.to_numpy(),
                           'flux':frame.sector_median.to_numpy()*np.exp(-correction)})
    groups = values.groupby('tic_id')
    cv = 100*groups.flux.std(ddof=0)/groups.flux.mean()
    return cv[groups.sector.nunique() >= 2]


def prepare():
    if (ROOT/'prepared.json').exists():
        return
    ROOT.mkdir(exist_ok=True)
    if (ROOT/'protocol.json').exists():
        raise RuntimeError('Archive the incomplete preparation protocol before retrying')
    reserve()
    columns = names()
    assert len(columns) == len(set(columns)) == 120
    assert not set(columns) & {'sector_median', 'gaia_log_anchor', 'source_row', 'tic_id', 'systematic_score'}
    saved = read(BASE/'prepared.json')
    protected = [BASE/'features.parquet', BASE/'cohort_manifest.parquet', BASE/'reference_bank.parquet',
                 BASE/'freeze.json', BASE/'physical/complete.json', BASE/'physical/imputation.pkl',
                 BASE/'benchmark_predictions.parquet', LABELS, Path('research_defaults.json'),
                 *BASE.glob('color_fold_*.pkl'), *BASE.glob('physical/round_*.pkl')]
    hashes = {str(p):sha(p) for p in protected}
    assert hashes[str(BASE/'features.parquet')] == saved['features_sha256']
    assert hashes[str(BASE/'cohort_manifest.parquet')] == saved['manifest_sha256']
    assert hashes[str(BASE/'reference_bank.parquet')] == saved['references_sha256']
    assert hashes[str(LABELS)] == saved['input_sha256'][str(LABELS)]
    protocol = dict(created_utc=pd.Timestamp.now(tz='UTC').isoformat(), seed=SEED,
        features=columns, flow_config=CONFIG, training_rows=457123, training_stars=36507,
        target='log(sector_median / pred_knn) - gaia_log_anchor', target_family='gaia_rp',
        reference='Exact HGB pred_knn and all 120 prepared predictors; not legacy CNF knn20 prior.',
        anchor='Exact saved nested training-only Gaia RP/color transfer; no per-star target centering for available anchors.',
        missing_anchor='Exact HGB relative fallback only where anchor missing; counts reported. No fitted training rows lack anchors.',
        preprocessing='Exact HGB training median imputation, then global training-only mean/std scaling; no feature clipping.',
        loss='Conditional negative log likelihood; HGB uses absolute-error boosting. Same target/data, different training objectives and capacities.',
        correction='pred_knn * exp(conditional median residual); beta=1, no correction clipping or calibration offsets.',
        optimizer=dict(name='Adam', learning_rate=3e-4, weight_decay=1e-5, gradient_norm_clip=5),
        epochs_max=MAX_EPOCHS, batch_size=BATCH, device='cpu', threads=4,
        device_choice='Synthetic 2048-row timing: CPU ~0.010 s/update, MPS ~0.041 s/update; choose CPU, no outcome-based choice.',
        selection='Minimum individually TARS-qualified calibration median per-star CV; earliest epoch on ties. Stop after 12 epochs without >1e-6 percentage-point improvement.',
        scheduler='ReduceLROnPlateau on anchored qualified-calibration NLL, patience=5, factor=0.5.',
        secondary_checkpoint='Minimum anchored qualified-calibration NLL saved as diagnostic, not selected on confirmation/test.',
        evaluation='All retained and individually systematic_score >0.95 sectors, with attrition, paired whole-star bootstrap, and Gaia-anchor residuals.',
        holdout_status='Historically examined development cohorts, not pristine holdouts.',
        scope='One matched full-data seed; no proof of architecture superiority or long-baseline signal preservation.',
        downloads=0, production_promotion=False, input_sha256=hashes, script_sha256=sha(__file__))
    save(ROOT/'protocol.json', protocol)
    (ROOT/'source_snapshot').mkdir(exist_ok=True)
    shutil.copy2(__file__, ROOT/'source_snapshot'/Path(__file__).name)
    f = pd.read_parquet(BASE/'features.parquet', columns=list(dict.fromkeys(columns+META)))
    assert f.source_row.is_unique
    manifest = pd.read_parquet(BASE/'cohort_manifest.parquet')
    for col in manifest:
        # The frozen HGB feature table casts float metadata to float32 after
        # saving the manifest. Compare in its actual stored precision.
        expected = manifest[col].astype(f[col].dtype) if pd.api.types.is_float_dtype(f[col]) else manifest[col]
        np.testing.assert_array_equal(f[col], expected)
    trmask = f.fit_row.to_numpy()
    cmask = f.evaluation_group.eq('calibration').to_numpy()
    assert trmask.sum() == 457123 and f.loc[trmask, 'tic_id'].nunique() == 36507
    assert f.loc[trmask, 'role'].eq('train').all()
    assert f.loc[trmask, 'systematic_score'].gt(.95).all()
    assert f.loc[trmask, 'has_gaia_anchor'].all()
    train_groups = set(f.loc[trmask, 'physical_group_id'])
    assert not train_groups & set(f.loc[f.role.ne('train'), 'physical_group_id'])
    bank = pd.read_parquet(BASE/'reference_bank.parquet')
    np.testing.assert_array_equal(np.sort(bank.source_row), np.sort(f.loc[trmask, 'source_row']))
    for fold in range(5):
        refs = set(bank.loc[bank.reference_fold.ne(fold), 'physical_group_id'])
        queries = set(f.loc[f.reference_fold.eq(fold), 'physical_group_id'])
        assert not refs & queries
    assert all(a['query_physical_overlap'] == a['nontraining_reference_rows'] == a['nonqualified_reference_rows'] == a['color_query_overlap'] == 0 for a in saved['audit'])
    del bank, manifest
    tr = f.loc[trmask, META].reset_index(drop=True)
    ca = f.loc[cmask, META].reset_index(drop=True)
    tr['sector_median'] = labels(tr)
    ca['sector_median'] = labels(ca)
    impute = joblib.load(BASE/'physical/imputation.pkl').reindex(columns)
    means, stds, missing_counts = {}, {}, {}
    x = np.empty((len(tr), 120), np.float32)
    cx = np.empty((len(ca), 120), np.float32)
    for j, col in enumerate(columns):
        v = f[col].to_numpy(); a = v[trmask]; b = v[cmask]
        median = float(pd.Series(a[np.isfinite(a)]).median()) if np.isfinite(a).any() else 0.
        assert median == impute[col], col
        a = np.where(np.isfinite(a), a, median).astype(np.float64)
        b = np.where(np.isfinite(b), b, median).astype(np.float64)
        mean = float(a.mean()); std = float(a.std(ddof=0))
        if std == 0: std = 1.
        means[col] = mean; stds[col] = std
        missing_counts[col] = int((~np.isfinite(v[trmask])).sum())
        x[:,j] = (a-mean)/std; cx[:,j] = (b-mean)/std
    assert np.isfinite(x).all() and np.isfinite(cx).all()
    y = target(tr, tr.sector_median.to_numpy()); cy = target(ca, ca.sector_median.to_numpy())
    ym = float(np.median(y)); ys = float(np.std(y))
    assert ys > 0 and np.isfinite(y).all()
    probe = tr.head(100).copy(); original = target(probe, probe.sector_median.to_numpy())
    probe.gaia_log_anchor += .01
    np.testing.assert_allclose(target(probe, probe.sector_median.to_numpy())-original, -.01, atol=1e-12, rtol=0)
    np.testing.assert_allclose(target(tr.head(100),tr.sector_median.to_numpy()[:100]*1.05)-y[:100],np.log(1.05),atol=1e-12,rtol=0)
    prep = dict(features=columns, medians=impute, means=pd.Series(means), stds=pd.Series(stds), y_median=ym, y_std=ys)
    joblib.dump(prep, ROOT/'preprocessing.pkl')
    for name, value in [('train_x',x),('calibration_x',cx),('train_y',((y-ym)/ys).astype(np.float32)),('calibration_y',((cy-ym)/ys).astype(np.float32))]:
        np.save(ROOT/(name+'.npy'),value)
    tr.to_parquet(ROOT/'training_rows.parquet',index=False)
    ca.to_parquet(ROOT/'calibration_rows.parquet',index=False)
    save(ROOT/'prepared.json',dict(status='passed', training_rows=len(tr),training_stars=tr.tic_id.nunique(),
        calibration_rows=len(ca),calibration_stars=ca.tic_id.nunique(),
        qualified_calibration_rows=int(ca.systematic_score.gt(.95).sum()),
        missing_training_anchors=int((~tr.has_gaia_anchor).sum()),missing_calibration_anchors=int((~ca.has_gaia_anchor).sum()),
        imputation_matches_hgb=True, exact_hgb_feature_table_verified=True,
        physical_group_exclusions_verified=True, reference_and_color_folds_verified=True,
        gaia_anchor_and_flux_response_checks=True, training_missing_features=missing_counts,
        standardized_training_max_abs=float(np.abs(x).max()), input_sha256=hashes,
        artifact_sha256={p.name:sha(p) for p in ROOT.glob('*.npy')}, preprocessing_sha256=sha(ROOT/'preprocessing.pkl')))
    print(f'Preparation passed: {len(tr):,} training rows, {len(columns)} inputs; no missing training anchors.',flush=True)


def distribution_outputs(flow, x, y=None, median=True):
    nll = 0.; result = []
    flow.eval()
    with torch.no_grad():
        for start in range(0,len(x),4096):
            xx = torch.from_numpy(x[start:start+4096])
            dist = flow(xx)
            if y is not None:
                nll -= float(dist.log_prob(torch.from_numpy(y[start:start+4096,None])).sum())
            if median:
                value = dist.transform.inv(torch.zeros((len(xx),1))).numpy().reshape(-1)
                result.append(value)
    return nll/len(x), np.concatenate(result) if median else None


def train():
    if (ROOT/'selection.json').exists(): return
    torch.set_num_threads(4)
    torch.manual_seed(SEED); np.random.seed(SEED)
    x = np.load(ROOT/'train_x.npy'); y = np.load(ROOT/'train_y.npy')
    cx = np.load(ROOT/'calibration_x.npy'); cy = np.load(ROOT/'calibration_y.npy')
    ca = pd.read_parquet(ROOT/'calibration_rows.parquet')
    prep = joblib.load(ROOT/'preprocessing.pkl')
    qualified = ca.systematic_score.gt(.95).to_numpy()
    anchored = qualified & ca.has_gaia_anchor.to_numpy()
    flow = zuko.flows.NSF(**CONFIG)
    opt = torch.optim.Adam(flow.parameters(),lr=3e-4,weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt,patience=5,factor=.5)
    xx = torch.from_numpy(x); yy = torch.from_numpy(y[:,None])
    gen = torch.Generator().manual_seed(SEED)
    history=[]; best_cv=float('inf'); best_nll=float('inf'); stale=0; start_epoch=0
    if (ROOT/'latest.pt').exists():
        c = torch.load(ROOT/'latest.pt',weights_only=False)
        assert c['protocol_sha256'] == sha(ROOT/'protocol.json')
        flow.load_state_dict(c['state']);opt.load_state_dict(c['optimizer']);sched.load_state_dict(c['scheduler'])
        gen.set_state(c['shuffle_state']);torch.set_rng_state(c['torch_rng'])
        history=c['history'];best_cv=c['best_cv'];best_nll=c['best_nll'];stale=c['stale'];start_epoch=c['epoch']
    for epoch in range(start_epoch,MAX_EPOCHS):
        reserve(); start=time.monotonic();flow.train();total_loss=0.
        order=torch.randperm(len(xx),generator=gen)
        for offset in range(0,len(xx),BATCH):
            ids=order[offset:offset+BATCH]
            loss=-flow(xx[ids]).log_prob(yy[ids]).mean()
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(flow.parameters(),5);opt.step()
            total_loss+=float(loss.detach())*len(ids)
        nll,_=distribution_outputs(flow,cx[anchored],cy[anchored],median=False)
        _,med=distribution_outputs(flow,cx)
        logcor=np.log(ca.pred_knn.to_numpy())+med*prep['y_std']+prep['y_median']
        if not np.isfinite(logcor).all():raise RuntimeError('Nonfinite correction')
        cv=float(cv_values(ca[qualified],logcor[qualified]).median())
        allcv=float(cv_values(ca,logcor).median())
        sched.step(nll)
        improved=cv < best_cv-1e-6
        if improved:best_cv=cv;stale=0
        else:stale+=1
        entry=dict(epoch=epoch+1,training_nll=total_loss/len(xx),calibration_nll=nll,
                   calibration_qualified_cv_pct=cv,calibration_all_cv_pct=allcv,
                   learning_rate=opt.param_groups[0]['lr'],seconds=time.monotonic()-start)
        history.append(entry)
        model=dict(state=copy.deepcopy(flow.state_dict()),config=CONFIG,feature_cols=prep['features'],
                   preprocessing=prep,epoch=epoch+1,target='gaia_rp',prior='pred_knn',beta=1.,clip=None,
                   protocol_sha256=sha(ROOT/'protocol.json'),metrics=entry)
        if improved:checkpoint(ROOT/'best_cv.pt',model)
        if nll < best_nll:
            best_nll=nll;checkpoint(ROOT/'best_nll.pt',model)
        checkpoint(ROOT/'latest.pt',dict(**model,optimizer=opt.state_dict(),scheduler=sched.state_dict(),
            shuffle_state=gen.get_state(),torch_rng=torch.get_rng_state(),history=history,
            best_cv=best_cv,best_nll=best_nll,stale=stale))
        pd.DataFrame(history).to_csv(ROOT/'history.csv',index=False)
        save(ROOT/'status.json',dict(status='training',pid=os.getpid(),epoch=epoch+1,max_epochs=MAX_EPOCHS,
             latest=entry,best_qualified_calibration_cv_pct=best_cv,stale_epochs=stale,
             free_disk_gib=reserve(),updated_utc=pd.Timestamp.now(tz='UTC').isoformat()))
        print(f'Epoch {epoch+1}/{MAX_EPOCHS}: train NLL {entry["training_nll"]:.5f}, calibration NLL {nll:.5f}, qualified CV {cv:.6f}%, best {best_cv:.6f}% ({entry["seconds"]:.1f}s)',flush=True)
        if stale>=PATIENCE:break
    best=torch.load(ROOT/'best_cv.pt',weights_only=False)
    save(ROOT/'selection.json',dict(selected='best_cv.pt',epoch=best['epoch'],metrics=best['metrics'],
        checkpoint_sha256=sha(ROOT/'best_cv.pt'),basis='Qualified calibration median per-star CV only',
        confirmation_or_test_used=False,epochs_completed=len(history),
        stop_reason='calibration patience' if stale>=PATIENCE else 'prespecified epoch budget',
        frozen_utc=pd.Timestamp.now(tz='UTC').isoformat()))
    print('Checkpoint selection frozen. Starting confirmation/development evaluation.',flush=True)


def evaluate():
    selection=read(ROOT/'selection.json')
    assert sha(ROOT/selection['selected'])==selection['checkpoint_sha256']
    torch.set_num_threads(4)
    c=torch.load(ROOT/selection['selected'],weights_only=False);prep=c['preprocessing']
    flow=zuko.flows.NSF(**c['config']);flow.load_state_dict(c['state']);flow.eval()
    f=pd.read_parquet(BASE/'features.parquet',columns=list(dict.fromkeys(names()+META)))
    f=f[f.role.ne('train')].reset_index(drop=True)
    x=np.empty((len(f),120),np.float32)
    for j,col in enumerate(names()):
        value=f[col].to_numpy().astype(float)
        value=np.where(np.isfinite(value),value,prep['medians'][col])
        x[:,j]=(value-prep['means'][col])/prep['stds'][col]
    q=f[META].copy();del f;gc.collect()
    q['sector_median']=labels(q)
    _,median=distribution_outputs(flow,x)
    q['log_correction_cnf']=np.log(q.pred_knn)+median*prep['y_std']+prep['y_median']
    bench=pd.read_parquet(BASE/'benchmark_predictions.parquet')
    assert set(q.source_row)==set(bench.source_row)
    bench=bench.set_index('source_row').loc[q.source_row]
    np.testing.assert_array_equal(q.sector_median,bench.sector_median)
    q['log_correction_hgb']=bench.log_correction_physical.to_numpy()
    q['log_correction_raw']=0.
    assert np.isfinite(q.log_correction_cnf).all()
    # Independent conditional-median check: transforming predicted medians gives z=0.
    with torch.no_grad():
        d=flow(torch.from_numpy(x[:256]));m=d.transform.inv(torch.zeros((256,1)))
        np.testing.assert_allclose(d.transform(m).numpy(),0,atol=2e-5,rtol=0)
    # A fixed-feature 5% signal survives multiplication exactly; not an upstream injection.
    factor=np.exp(q.log_correction_cnf.to_numpy())
    np.testing.assert_allclose((q.sector_median.to_numpy()*1.05/factor)/(q.sector_median.to_numpy()/factor),1.05,rtol=1e-12)
    q.to_parquet(ROOT/'predictions.parquet',index=False)
    metrics=[];parts=[];paired=[];rng=np.random.default_rng(SEED)
    for group,g in q.groupby('evaluation_group'):
        for view,mask in [('all',np.ones(len(g),bool)),('qualified',g.systematic_score.gt(.95).to_numpy())]:
            v=g[mask]; series={}
            for model in ['raw','hgb','cnf']:
                correction=v['log_correction_'+model].to_numpy()
                cv=cv_values(v,correction);series[model]=cv
                a=v.has_gaia_anchor.to_numpy()
                err=np.log(v.sector_median.to_numpy()[a])-correction[a]-v.gaia_log_anchor.to_numpy()[a]
                metrics.append(dict(group=group,view=view,model=model,rows=len(v),stars=int(v.tic_id.nunique()),metric_stars=len(cv),
                    missing_anchors=int((~a).sum()),median_cv_pct=float(cv.median()),p90_cv_pct=float(cv.quantile(.9)),
                    median_abs_gaia_log_error=float(np.median(np.abs(err)))))
                parts.append(cv.rename('cv_pct').reset_index().assign(group=group,view=view,model=model))
            gains=(series['hgb']-series['cnf']).dropna().to_numpy()
            boots=np.array([np.median(rng.choice(gains,len(gains),replace=True)) for _ in range(2000)])
            paired.append(dict(group=group,view=view,stars=len(gains),median_gain_pp=float(np.median(gains)),
                ci95_low_pp=float(np.quantile(boots,.025)),ci95_high_pp=float(np.quantile(boots,.975)),
                fraction_improved=float(np.mean(gains>0))))
    pd.DataFrame(metrics).to_csv(ROOT/'metrics.csv',index=False)
    pd.concat(parts,ignore_index=True).to_parquet(ROOT/'per_star.parquet',index=False)
    pd.DataFrame(paired).to_csv(ROOT/'paired_comparisons.csv',index=False)
    measured=next(m for m in metrics if m['group']=='calibration' and m['view']=='qualified' and m['model']=='cnf')['median_cv_pct']
    np.testing.assert_allclose(measured,selection['metrics']['calibration_qualified_cv_pct'],rtol=0,atol=1e-7)
    protocol=read(ROOT/'protocol.json')
    for path,digest in protocol['input_sha256'].items():assert sha(path)==digest,path
    save(ROOT/'verification.json',dict(status='passed',training_rows=457123,training_stars=36507,features=120,
        exact_hgb_features_references_anchor=True,preflight='prepared.json',selection_before_evaluation=True,
        selected_checkpoint_reproduces_calibration=True,conditional_median_inverse_check=True,
        fixed_feature_flux_signal_check=True,upstream_signal_preservation_tested=False,
        historical_inputs_preserved=True,new_downloads=0,production_promoted=False,
        selection_sha256=sha(ROOT/'selection.json'),predictions_sha256=sha(ROOT/'predictions.parquet')))
    lines=['# Matched Gaia-RP CNF retrain','',
        'One CNF was trained on the exact physical HGB training rows, 120 inputs, reference calculations and Gaia RP target. All historical inputs were preserved; no acquisition occurred.','',
        f'457,123 qualified sectors from 36,507 training stars. Selected epoch {selection["epoch"]} of {selection["epochs_completed"]} completed epochs, using qualified calibration CV only. No missing training anchors; 21 calibration rows use the original explicitly flagged missing-anchor fallback.','',
        'The CNF fits a conditional distribution by negative log likelihood and uses its median as the correction. HGB uses absolute-error boosting. This matches data, inputs and target, but not objective or model capacity; one run cannot establish which architecture is generally better.','',
        '| Group | View | Input sectors | Metric stars | Raw CV (%) | HGB CV (%) | CNF CV (%) |',
        '|---|---|---:|---:|---:|---:|---:|']
    tab=pd.DataFrame(metrics)
    for (group,view),part in tab.groupby(['group','view']):
        by=part.set_index('model');r=part.iloc[0]
        lines.append(f'| {group} | {view} | {r.rows:,} | {r.metric_stars:,} | {by.loc["raw","median_cv_pct"]:.6f} | {by.loc["hgb","median_cv_pct"]:.6f} | {by.loc["cnf","median_cv_pct"]:.6f} |')
    lines+=['','Qualified means each sector has systematic_score >0.95. Counts above show attrition; all/qualified samples are not interchangeable. Positive paired gain in paired_comparisons.csv means CNF lowers per-star CV. Intervals resample whole stars and do not include seed/model-selection uncertainty.','',
        'These are historically examined development cohorts, not pristine holdouts. Quiet-star scatter and Gaia residuals do not establish a photometric floor or long-baseline signal preservation. The fixed-feature signal check is only algebraic; extraction/PDC responses have not been rerun. No model is automatically promoted.','',
        '[Protocol](protocol.json) · [Selection](selection.json) · [Metrics](metrics.csv) · [Paired comparisons](paired_comparisons.csv) · [Verification](verification.json)','']
    (ROOT/'REPORT.md').write_text('\n'.join(lines))
    save(ROOT/'status.json',dict(status='complete',selected_epoch=selection['epoch'],epochs_completed=selection['epochs_completed'],
         report=str(ROOT/'REPORT.md'),updated_utc=pd.Timestamp.now(tz='UTC').isoformat()))
    print(tab[['group','view','model','median_cv_pct']].to_string(index=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','train','evaluate','run'],default='run',nargs='?')
    args=parser.parse_args()
    try:
        if args.stage in ['prepare','run']:prepare()
        if args.stage in ['train','run']:train()
        if args.stage in ['evaluate','run']:evaluate()
    except Exception as error:
        if ROOT.exists():save(ROOT/'failure.json',dict(error=repr(error),utc=pd.Timestamp.now(tz='UTC').isoformat()))
        raise
