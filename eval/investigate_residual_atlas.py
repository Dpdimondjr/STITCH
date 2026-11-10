"""Validation-only time, sky, detector, and reference-support diagnostics.

Secondary models are fitted out of fold by whole TIC. They are development
diagnostics, not new independent test performance. All figures retain the
original cohort; display clipping never excludes observations from metrics.
"""
import os
os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/stitch_mpl')
import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord, BarycentricTrueEcliptic
import astropy.units as u
from scipy.stats import spearmanr, binned_statistic_2d
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from hgb_reference_repair import sha256

SEED = 20261012


def save_json(path, obj):
    def clean(x):
        if isinstance(x, dict): return {str(k): clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple, np.ndarray)): return [clean(v) for v in x]
        if isinstance(x, (np.integer,)): return int(x)
        if isinstance(x, (float, np.floating)): return float(x) if np.isfinite(x) else None
        if isinstance(x, (np.bool_,)): return bool(x)
        return x
    Path(path).write_text(json.dumps(clean(obj), indent=2, allow_nan=False)+'\n')


def log(message): print(time.strftime('%H:%M:%S'), message, flush=True)


def assign_folds(ids, n=5, seed=SEED):
    unique = np.sort(np.unique(ids)); rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unique)
    return pd.Series(np.arange(len(shuffled)) % n, index=shuffled)


def temporal_statistics(values, times):
    """Each row is one possible time ordering; steps need >=2 points each side."""
    a = np.atleast_2d(values).astype(float)
    a = a-a.mean(axis=1, keepdims=True)
    ss = np.sum(a*a, axis=1)
    ss = np.maximum(ss, 1e-20)
    t = np.asarray(times, float); t = t-t.mean()
    trend = (a@t)**2/(ss*max(np.sum(t*t), 1e-20))
    k = np.arange(2, a.shape[1]-1)
    prefix = np.cumsum(a, axis=1)[:, k-1]
    step = np.max(prefix**2*(1/k+1/(a.shape[1]-k)), axis=1)/ss
    adjacent = np.sum(a[:, :-1]*a[:, 1:], axis=1)/ss
    return np.column_stack([trend, step, adjacent])


def star_cv(frame, log_adjustment=None):
    z = frame[['tic_id', 'corrected_flux']].copy()
    if log_adjustment is not None: z.corrected_flux *= np.exp(-np.asarray(log_adjustment))
    g = z.groupby('tic_id').corrected_flux
    return 100*np.sqrt(g.var(ddof=0))/g.mean()


def paired_gain(a, b, seed=SEED, draws=1000):
    z = pd.concat([a.rename('a'), b.rename('b')], axis=1).dropna().to_numpy()
    rng = np.random.default_rng(seed)
    gains = [np.median(z[j, 0])-np.median(z[j, 1])
             for j in rng.integers(0, len(z), (draws, len(z)))]
    return {'stars': len(z), 'before_cv_pct': np.median(z[:, 0]),
            'after_cv_pct': np.median(z[:, 1]),
            'median_cv_gain_pp': np.median(z[:, 0])-np.median(z[:, 1]),
            'conditional_star_bootstrap95_pp': np.quantile(gains, [.025, .975]),
            'fraction_stars_improve': np.mean(z[:, 1] < z[:, 0])}


def build_frame(root):
    frame = pd.read_parquet('eval/repaired_residual_diagnosis_20261010/validation_observations.parquet')
    cols = ['source_row','col','row','ra','dec','knn5_radius','knn20_radius','knn50_radius',
            'ref_count','numcont','bp_rp','median_sap_bkg','scatter_flag_frac',
            'focal_plane_temp','focal_plane_temp_std','jitter_rms','pdc_totp','pdc_corp',
            'dev_crowdsap_loo','dev_jitter_rms_loo','dev_pdcvar_loo','dev_fptemp_loo']
    feat = pd.read_parquet(root/'clean_features.parquet', columns=cols)
    frame = frame.merge(feat, on='source_row', validate='one_to_one').sort_values(['tic_id','sector']).reset_index(drop=True)
    assert frame.role.eq('validation').all() and not frame.duplicated(['tic_id','sector']).any()
    coord = frame.groupby('tic_id')[['ra','dec']].first()
    sky = SkyCoord(coord.ra.to_numpy()*u.deg, coord.dec.to_numpy()*u.deg)
    ecl = sky.transform_to(BarycentricTrueEcliptic())
    coord['ecl_lon'] = ecl.lon.deg; coord['ecl_lat'] = ecl.lat.deg
    coord['gal_lon'] = sky.galactic.l.deg; coord['gal_lat'] = sky.galactic.b.deg
    frame = frame.join(coord[['ecl_lon','ecl_lat','gal_lon','gal_lat']], on='tic_id')
    frame['abs_ecl_lat'] = frame.ecl_lat.abs(); frame['abs_gal_lat'] = frame.gal_lat.abs()
    frame['n_observed'] = frame.groupby('tic_id').sector.transform('size')
    frame['log_residual'] = np.log(frame.corrected_flux)-frame.groupby('tic_id').corrected_flux.transform(lambda x: np.log(x).mean())
    frame['detector'] = (frame.cam-1)*4+frame.ccd
    frame['fold'] = frame.tic_id.map(assign_folds(frame.tic_id)).astype(int)
    frame['edge_distance'] = np.minimum.reduce([frame.col-44, 2092-frame.col, frame.row, 2048-frame.row])
    for c in ['crowdsap','flfrcsap','bkg_rms','median_sap_bkg','knn20_radius','jitter_rms','perr_col','perr_row']:
        frame[c+'_centered'] = frame[c]-frame.groupby('tic_id')[c].transform('mean')
    return frame


def trajectory_diagnosis(frame, stars, out):
    rng = np.random.default_rng(SEED); rows = []; draws = 399
    for tic, g in frame.groupby('tic_id', sort=False):
        if len(g) < 8: continue
        y = g.log_residual.to_numpy(); t = g.sector.to_numpy()
        perm = y[np.argsort(rng.random((draws, len(y))), axis=1)]
        actual = temporal_statistics(y, t)[0]; null = temporal_statistics(perm, t)
        row = {'tic_id': tic, 'n': len(g)}
        for j, name in enumerate(['linear_trend','step','adjacent']):
            row[name] = actual[j]
            row[name+'_permutation_p'] = (1+np.sum(null[:, j] >= actual[j]))/(draws+1)
            row[name+'_null_median'] = np.median(null[:, j])
        i = np.argmax(abs(y))
        row.update(maximum_sector=int(t[i]), maximum_abs_residual_pct=100*abs(y[i]),
                   maximum_ss_fraction=y[i]**2/max(np.sum(y*y),1e-20))
        rows.append(row)
    shape = pd.DataFrame(rows).set_index('tic_id')
    shape.to_csv(out/'temporal_order.csv')
    summary = {'eligible_stars': len(shape), 'permutations_per_star': draws,
               'fraction_p_le_001': {n: float((shape[n+'_permutation_p'] <= .01).mean())
                                     for n in ['linear_trend','step','adjacent']},
               'mean_adjacent_statistic': shape.adjacent.mean(),
               'mean_permuted_adjacent_median': shape.adjacent_null_median.mean(),
               'limitations': 'Permutation preserves each observed value distribution and times, but not time-dependent noise or detector changes; temporal order does not prove intrinsic variability.'}
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.7), constrained_layout=True)
    for ax, name, title in zip(axes, ['linear_trend','step','adjacent'], ['Linear drift','Best ordered step','Adjacent-sector persistence']):
        ax.hist(shape[name+'_permutation_p'], bins=np.linspace(0,1,21), color='#457b9d')
        ax.axhline(len(shape)/20, color='#b54c3c', ls='--', label='Uniform-null expectation')
        ax.set(xlabel='Within-star permutation p', ylabel='Stars', title=title)
    axes[0].legend(fontsize=8); fig.suptitle('Does the order of sectors matter?  Stars with at least 8 observations')
    fig.savefig(out/'temporal_order.png', dpi=170); plt.close(fig)
    selected = []; categories = []
    source = stars.join(shape, rsuffix='_order')
    for category, candidates in [
        ('Isolated excursion', source.sort_values('maximum_sector_ss_fraction', ascending=False)),
        ('Ordered step / trend', source[source.step_permutation_p<=.01].sort_values('corrected_cv_pct', ascending=False)),
        ('Typical residual', source.assign(distance=(source.corrected_cv_pct-source.corrected_cv_pct.median()).abs()).sort_values('distance'))]:
        for tic in candidates.index:
            if tic not in selected and source.loc[tic,'n_sectors']>=8:
                selected.append(tic); categories.append(category)
                if categories.count(category)==3: break
    fig, axes = plt.subplots(3, 3, figsize=(14, 9), constrained_layout=True)
    for ax, tic, cat in zip(axes.ravel(), selected, categories):
        g=frame[frame.tic_id==tic]
        raw=100*(g.sector_median/g.sector_median.mean()-1)
        ax.plot(g.sector,raw,'o-',color='.72',ms=3,lw=.8,label='Raw PDCSAP')
        ax.plot(g.sector,g.signed_residual_pct,'o-',color='#226b8c',ms=4,lw=1,label='After STITCH')
        ax.axhline(0,color='.8',lw=.6)
        ax.set(title=f'{cat}\nTIC {tic}; CV {source.loc[tic,"corrected_cv_pct"]:.2f}%',xlabel='Sector number',ylabel='Relative flux (%)')
    axes[0,0].legend(fontsize=8); fig.suptitle('Examples selected by stated shape criteria; connecting lines span observing gaps')
    fig.savefig(out/'star_trajectories.png',dpi=170);plt.close(fig)
    save_json(out/'trajectory_examples.json', [{'tic_id':int(t),'selection':c} for t,c in zip(selected,categories)])
    return summary


def risk_models(frame, stars, out):
    aggregations = {c:'median' for c in ['ref_count','knn5_radius','knn20_radius','knn50_radius',
        'abs_ecl_lat','ecl_lat','ecl_lon','abs_gal_lat','gal_lon','numcont','contratio','flfrcsap','bkg_rms',
        'median_sap_bkg','edge_distance','jitter_rms','pdcvar','pdc_totp','bp_rp']}
    stars = stars.join(frame.groupby('tic_id').agg(aggregations),rsuffix='_feature')
    stars['sector_span'] = frame.groupby('tic_id').sector.max()-frame.groupby('tic_id').sector.min()
    stars['detectors_seen'] = frame.groupby('tic_id').detector.nunique()
    stars['fold'] = stars.index.map(assign_folds(stars.index))
    basic=['tmag','n_sectors','crowdsap']
    groups={'brightness_count_crowding':basic,
        'plus_reference_coverage':basic+['ref_count','knn5_radius','knn20_radius','knn50_radius'],
        'plus_sky':basic+['abs_ecl_lat','ecl_lat','ecl_lon','abs_gal_lat','gal_lon'],
        'plus_scene':basic+['contratio','numcont','flfrcsap','bp_rp'],
        'plus_history_detector':basic+['sector_span','detectors_seen','edge_distance'],
        'plus_background_pdc':basic+['bkg_rms','median_sap_bkg','jitter_rms','pdcvar','pdc_totp']}
    good=stars.corrected_cv_pct>0
    stars=stars[good].copy();y=np.log(stars.corrected_cv_pct.to_numpy())
    predictions={};metrics={}
    for name,cols in groups.items():
        pred=np.full(len(stars),np.nan)
        for fold in range(5):
            tr=stars.fold!=fold; va=~tr
            model=HistGradientBoostingRegressor(loss='squared_error',max_iter=100,max_leaf_nodes=7,
                min_samples_leaf=60,l2_regularization=5,learning_rate=.05,early_stopping=False,random_state=SEED)
            model.fit(stars.loc[tr,cols],y[tr]);pred[va]=model.predict(stars.loc[va,cols])
        predictions[name]=pred
        metrics[name]={'oof_log_cv_r2':r2_score(y,pred),'oof_log_cv_mae':mean_absolute_error(y,pred), 'features':cols}
        log(f'Risk {name}: OOF R2 {metrics[name]["oof_log_cv_r2"]:.4f}')
    stars['risk_adjusted_log_cv'] = y-predictions['brightness_count_crowding']
    for name, pred in predictions.items():stars['oof_'+name]=pred
    stars.to_parquet(out/'star_risk.parquet')
    strata=[]
    configs=[('tmag',np.arange(7,14)),('n_sectors',[1,4,8,16,32,100]),
             ('crowdsap',[0,.8,.9,.95,.98,1.01]),('knn20_radius',[0,50,100,150,200,300,500,3000]),
             ('abs_ecl_lat',[0,15,30,45,60,75,90]),('abs_gal_lat',[0,10,20,30,45,60,90])]
    fig,axes=plt.subplots(2,3,figsize=(14,8),constrained_layout=True)
    for ax,(column,bins) in zip(axes.ravel(),configs):
        groups2=stars.groupby(pd.cut(stars[column],bins,include_lowest=True),observed=True)
        x=[];cv=[]; adjusted=[];labels=[]
        for interval,g in groups2:
            if len(g)<10:continue
            strata.append({'variable':column,'bin':str(interval),'stars':len(g),
                'median_cv_pct':g.corrected_cv_pct.median(),'risk_adjusted_log_cv_median':g.risk_adjusted_log_cv.median()})
            labels.append(f'{interval.left:g}–{interval.right:g}\n(n={len(g)})')
            cv.append(g.corrected_cv_pct.median()); adjusted.append(100*np.expm1(g.risk_adjusted_log_cv.median()))
        x=np.arange(len(cv));ax.plot(x,cv,'o-',color='#226b8c')
        ax.set_xticks(x,labels,rotation=30,ha='right',fontsize=8)
        ax.set(xlabel=column.replace('_',' '),ylabel='Median per-star CV (%)')
        twin=ax.twinx();twin.plot(x,adjusted,'s--',color='#b55338',alpha=.85)
        twin.axhline(0,color='#b55338',lw=.5,alpha=.4);twin.set_ylabel('CV relative to risk prediction (%)',color='#b55338',fontsize=9)
    fig.suptitle('Unadjusted CV (blue) and remaining risk after brightness, crowding, sector count (orange)')
    fig.savefig(out/'coverage_and_sky_strata.png',dpi=170);plt.close(fig)
    pd.DataFrame(strata).to_csv(out/'star_strata.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(13,4.5),subplot_kw={'projection':'mollweide'},constrained_layout=True)
    for ax,lon,lat,title in [(axes[0],'ecl_lon','ecl_lat','Ecliptic coordinates'),(axes[1],'gal_lon','abs_gal_lat','Galactic coordinates')]:
        latitude=stars[lat] if lat!='abs_gal_lat' else frame.groupby('tic_id').gal_lat.first().reindex(stars.index)
        xx=np.deg2rad((stars[lon]+180)%360-180)
        sc=ax.scatter(-xx,np.deg2rad(latitude),c=stars.risk_adjusted_log_cv,s=8,cmap='coolwarm',vmin=-.7,vmax=.7,alpha=.75,rasterized=True)
        tick=np.arange(-150,151,30)
        ax.set_xticks(np.deg2rad(tick),[f'{(-v)%360}°' for v in tick])
        ax.grid(alpha=.3);ax.set_title(title)
    fig.colorbar(sc,ax=list(axes),label='log(CV / out-of-fold risk prediction)',shrink=.7)
    fig.suptitle('Same validation stars; sky pattern adjusted for brightness, crowding, and number of sectors')
    fig.savefig(out/'sky_maps.png',dpi=170);plt.close(fig)
    return stars,metrics


def field_prediction(frame, keys):
    pred=np.zeros(len(frame));support=np.zeros(len(frame))
    for fold in range(5):
        a=frame[frame.fold!=fold];b=frame[frame.fold==fold]
        stats=a.groupby(keys).log_residual.agg(['sum','count'])
        # Fixed 30-observation shrinkage to zero; no strength optimization.
        stats['prediction']=stats['sum']/(stats['count']+30)
        joined=b[keys].merge(stats[['prediction','count']],left_on=keys,right_index=True,how='left',sort=False)
        pred[b.index]=joined.prediction.fillna(0).to_numpy()
        support[b.index]=joined['count'].fillna(0).to_numpy()
    return pred,support


def detector_diagnosis(frame, out):
    table=frame.groupby(['sector','detector']).agg(stars=('tic_id','nunique'),
        signed_median_pct=('signed_residual_pct','median'),absolute_median_pct=('absolute_residual_pct','median'))
    table.to_csv(out/'sector_detector_cells.csv')
    fig,axes=plt.subplots(3,1,figsize=(15,9),constrained_layout=True)
    for ax,col,cmap,lims,title in zip(axes,['signed_median_pct','absolute_median_pct','stars'],['RdBu_r','magma','viridis'],[(-1,1),(0,1.5),(0,300)],['Signed residual median (%)','Absolute residual median (%)','Stars per sector / detector']):
        p=table[col].unstack('sector').reindex(range(1,17))
        if col!='stars':p=p.where(table.stars.unstack('sector').reindex(range(1,17))>=10)
        im=ax.imshow(p,aspect='auto',origin='lower',cmap=cmap,vmin=lims[0],vmax=lims[1])
        ticks=np.arange(0,len(p.columns),5);ax.set_xticks(ticks,p.columns[ticks]);ax.set_yticks(np.arange(16),[f'C{(n-1)//4+1}/{(n-1)%4+1}' for n in range(1,17)],fontsize=8)
        ax.set(title=title,xlabel='Sector',ylabel='Camera / CCD');fig.colorbar(im,ax=ax,pad=.01)
    fig.suptitle('Signed patterns, difficult fields, and coverage are different questions; blank metric cells have <10 stars')
    fig.savefig(out/'sector_detector_maps.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(4,4,figsize=(12,11),constrained_layout=True)
    spatial=[]
    for detector,ax in enumerate(axes.ravel(),1):
        g=frame[frame.detector==detector]
        bins=[np.linspace(44,2092,9),np.linspace(0,2048,9)]
        statistic,xe,ye,_=binned_statistic_2d(g.col,g.row,g.absolute_residual_pct,statistic='median',bins=bins)
        count,_,_,_=binned_statistic_2d(g.col,g.row,g.absolute_residual_pct,statistic='count',bins=bins)
        statistic[count<30]=np.nan
        im=ax.pcolormesh(xe,ye,statistic.T,cmap='magma',vmin=0,vmax=1.5,shading='flat')
        ax.set(title=f'Camera {(detector-1)//4+1}, CCD {(detector-1)%4+1}',xlabel='Catalog column (pixel)',ylabel='Catalog row (pixel)')
        for i,j in np.ndindex(statistic.shape):spatial.append({'detector':detector,'x_bin':i,'y_bin':j,'observations':count[i,j],'median_absolute_pct':statistic[i,j]})
    fig.colorbar(im,ax=list(axes.ravel()),label='Median absolute residual (%)',shrink=.6)
    fig.suptitle('Detector position pooled across sectors; unadjusted, cells with at least 30 observations')
    fig.savefig(out/'detector_position_maps.png',dpi=170);plt.close(fig)
    pd.DataFrame(spatial).to_csv(out/'detector_position_cells.csv',index=False)
    baseline=star_cv(frame);metrics={}
    for resolution in [1,2,4]:
        frame['x_bin']=np.clip(((frame.col-44)/2048*resolution).astype(int),0,resolution-1)
        frame['y_bin']=np.clip((frame.row/2048*resolution).astype(int),0,resolution-1)
        keys=['sector','detector']+(['x_bin','y_bin'] if resolution>1 else [])
        pred,support=field_prediction(frame,keys)
        metrics[f'{resolution}x{resolution}']=paired_gain(baseline,star_cv(frame,.5*pred))
        metrics[f'{resolution}x{resolution}'].update(median_support=np.median(support),
            oof_signed_r2=r2_score(frame.log_residual,pred),fixed_strength=.5)
        frame[f'field_prediction_{resolution}']=pred
    return metrics


def signed_response_models(frame, out):
    """Small prespecified models: can metadata predict signed residuals on new TICs?"""
    common=['sector','cam','ccd','col','row','tmag','crowdsap','n_observed']
    groups={'location':common,
        'coverage':common+['knn5_radius','knn20_radius','knn50_radius','ref_count','knn20_radius_centered'],
        'scene_response':common+['flfrcsap','contratio','numcont','crowdsap_centered','flfrcsap_centered','bp_rp'],
        'background_pointing':common+['bkg_rms','median_sap_bkg','bkg_rms_centered','median_sap_bkg_centered',
             'jitter_rms_centered','perr_col_centered','perr_row_centered','focal_plane_temp','pdcvar'],
        'sky':common+['ecl_lat','ecl_lon','gal_lat','gal_lon']}
    baseline=star_cv(frame);results={}
    for name,columns in groups.items():
        pred=np.zeros(len(frame))
        for fold in range(5):
            tr=frame.fold!=fold;va=~tr
            model=HistGradientBoostingRegressor(loss='squared_error',max_iter=100,max_leaf_nodes=15,
                min_samples_leaf=100,l2_regularization=10,learning_rate=.04,early_stopping=False,random_state=SEED)
            model.fit(frame.loc[tr,columns],frame.loc[tr,'log_residual'],sample_weight=1/frame.loc[tr,'n_observed'])
            pred[va]=model.predict(frame.loc[va,columns])
        results[name]=paired_gain(baseline,star_cv(frame,.5*pred))
        results[name].update(oof_signed_r2=r2_score(frame.log_residual,pred),features=columns,fixed_strength=.5)
        frame['response_'+name]=pred
        log(f'Signed {name}: CV gain {results[name]["median_cv_gain_pp"]:.5f} pp')
    return results


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',default='eval/hgb_reference_repair_20261010')
    p.add_argument('--out',default='eval/residual_investigation_20261011/atlas')
    a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    frame=build_frame(Path(a.root));log(f'Loaded {len(frame)} validation observations')
    stars=pd.read_parquet('eval/repaired_residual_diagnosis_20261010/validation_per_star.parquet').set_index('tic_id')
    temporal=trajectory_diagnosis(frame,stars,out);log('Temporal order and example plots complete')
    stars,risk=risk_models(frame,stars,out)
    fields=detector_diagnosis(frame,out);log('Detector maps and cross-fitted fields complete')
    signed=signed_response_models(frame,out)
    frame.to_parquet(out/'observations.parquet',index=False)
    result={'model':'clean_deviation seed 42','role':'validation','stars':frame.tic_id.nunique(),
        'observations':len(frame),'baseline_cv_pct':star_cv(frame).median(),
        'temporal_order':temporal,'risk_models':risk,'common_field_predictions':fields,'signed_models':signed,
        'script_sha256':sha256(__file__),
        'limitations':['Exploratory validation diagnostics; not independent test claims.',
                       'Five whole-TIC folds prevent fitting a star with its own residuals.',
                       'Conditional bootstrap does not include refitting or exploration uncertainty.',
                       'Sky and detector maps can reflect selection and covariate differences.',
                       'Temporal order does not distinguish instrumental from astrophysical change.']}
    save_json(out/'results.json',result)
    lines=['# Remaining-residual atlas','',f'{len(frame):,} observations of {frame.tic_id.nunique():,} validation stars; baseline CV {star_cv(frame).median():.6f}%.','',
        '## Temporal order','',json.dumps(temporal,indent=2),'','## Signed correction diagnostics','',
        'Fixed half-strength corrections predicted on other TICs. Positive gain means lower median CV. These are development experiments.','',
        '| Model | Median CV gain (pp) | Conditional 95% interval | Signed R² |','|---|---:|---|---:|']
    for name,r in {**{'field_'+k:v for k,v in fields.items()},**signed}.items():
        ci=r['conditional_star_bootstrap95_pp'];lines.append(f'| {name} | {r["median_cv_gain_pp"]:.6f} | [{ci[0]:.6f}, {ci[1]:.6f}] | {r["oof_signed_r2"]:.4f} |')
    lines+=['','## Difficulty prediction','', '| Features | OOF log-CV R² | OOF MAE |','|---|---:|---:|']
    for name,r in risk.items():lines.append(f'| {name} | {r["oof_log_cv_r2"]:.4f} | {r["oof_log_cv_mae"]:.4f} |')
    lines+=['','## Figures','']+[f'- [{n.replace("_"," ")}]({n}.png)' for n in ['star_trajectories','temporal_order','sector_detector_maps','detector_position_maps','coverage_and_sky_strata','sky_maps']]
    lines+=['','## Limits','']+['- '+s for s in result['limitations']]
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n');log('Atlas complete')


if __name__=='__main__':main()
