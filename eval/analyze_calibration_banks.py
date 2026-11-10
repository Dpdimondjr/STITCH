"""Metrics, paired uncertainty, disagreement and diagnostic plots for bank study."""
import os
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/stitch_mpl')
import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from hgb_reference_repair import per_star,write_json,sha256
from calibration_bank_study import OUT


def score(frame,correction,mask):
    return per_star(frame[mask],np.exp(np.asarray(correction)[mask]))


def paired_interval(a,b,seed=202610116):
    z=pd.concat([a.rename('a'),b.rename('b')],axis=1).dropna()
    rng=np.random.default_rng(seed);a=z.a.to_numpy();b=z.b.to_numpy()
    estimates=[]
    for _ in range(1200):
        ix=rng.integers(0,len(z),len(z));estimates.append(np.median(a[ix])-np.median(b[ix]))
    return dict(stars=len(z),gain_pp=float(np.median(a)-np.median(b)),
                gain95_pp=np.quantile(estimates,[.025,.975]).tolist(),
                improved=int((b<a).sum()),worsened=int((b>a).sum()),harms_gt_01pp=int((b-a>.1).sum()))


def bank_analysis(group):
    records=[];starrows=[];pairchecks=[]
    for rep in [0,1]:
        for size in ['eighth','quarter','half']:
            roots=[OUT/f'r{rep}_{size}_{bank}' for bank in ['A','B']]
            paths=[root/'base_r3_l255_i200/predictions.parquet' for root in roots]
            if not all((path.parent/'complete.json').exists() for path in paths):continue
            a,b=[pd.read_parquet(path) for path in paths]
            pd.testing.assert_frame_equal(a[['source_row','tic_id','sector']],b[['source_row','tic_id','sector']])
            sizes=[json.loads((root/'prepared.json').read_text())['training_stars'] for root in roots]
            pa=a.log_correction_r3.to_numpy();pb=b.log_correction_r3.to_numpy();pe=(pa+pb)/2
            for view in ['all','qualified']:
                mask=a.evaluation_group.eq(group).to_numpy()
                if view=='qualified':mask &= a.systematic_score.gt(.95).to_numpy()
                sa=score(a,pa,mask);sb=score(a,pb,mask);se=score(a,pe,mask)
                diff=pd.Series(100*(pa-pb)[mask],index=a.loc[mask,'tic_id']).groupby(level=0).std(ddof=0).reindex(se.index)
                raw=score(a,np.zeros(len(a)),mask)
                rec=dict(replicate=rep,size=size,view=view,group=group,bank_A_stars=sizes[0],bank_B_stars=sizes[1],
                         observations=int(mask.sum()),metric_stars=len(se),raw_cv_pct=float(raw.median()),
                         bank_A_cv_pct=float(sa.median()),bank_B_cv_pct=float(sb.median()),
                         ensemble_cv_pct=float(se.median()),disagreement_log_sd_pct=float(diff.median()),
                         disagreement_p90_pct=float(diff.quantile(.9)))
                rec['bank_A_to_ensemble']=paired_interval(sa,se)
                records.append(rec)
                s=pd.DataFrame(dict(bank_A=sa,bank_B=sb,ensemble=se,raw=raw,disagreement=diff)).reset_index()
                s['replicate']=rep;s['size']=size;s['view']=view;s['group']=group;starrows.append(s)
                # Algebraic covariance decomposition, descriptive only.
                rA=pd.Series(np.log(a.loc[mask,'sector_median'].to_numpy())-pa[mask],index=a.loc[mask,'tic_id'])
                rB=pd.Series(np.log(a.loc[mask,'sector_median'].to_numpy())-pb[mask],index=a.loc[mask,'tic_id'])
                rA=rA-rA.groupby(level=0).transform('mean');rB=rB-rB.groupby(level=0).transform('mean')
                vA=(rA*rA).groupby(level=0).mean();vB=(rB*rB).groupby(level=0).mean()
                cov=(rA*rB).groupby(level=0).mean();vd=((rA-rB)**2).groupby(level=0).mean()
                np.testing.assert_allclose(vA+vB-2*cov,vd,atol=1e-13)
                pairchecks.append(dict(replicate=rep,size=size,view=view,group=group,
                    shared_residual_covariance=float(cov.mean()),variance_A=float(vA.mean()),variance_B=float(vB.mean()),
                    residual_correlation=float(cov.mean()/np.sqrt(vA.mean()*vB.mean())),
                    difference_variance=float(vd.mean()),note='Equal-star centered second moments, not median CV; shared residual may be correctable.'))
    if not records:return
    write_json(OUT/f'bank_metrics_{group}.json',records)
    write_json(OUT/f'bank_covariance_{group}.json',pairchecks)
    table=pd.DataFrame([{k:v for k,v in r.items() if k!='bank_A_to_ensemble'} for r in records])
    table.to_csv(OUT/f'bank_metrics_{group}.csv',index=False)
    pd.concat(starrows).to_parquet(OUT/f'bank_per_star_{group}.parquet',index=False)
    fig,ax=plt.subplots(1,3,figsize=(15,4.6),constrained_layout=True)
    for view,color in [('all','#ad5941'),('qualified','#246b8b')]:
        for rep,style in [(0,'-'),(1,'--')]:
            g=table[(table.view==view)&(table.replicate==rep)].sort_values('bank_A_stars')
            ax[0].plot(g.bank_A_stars,(g.bank_A_cv_pct+g.bank_B_cv_pct)/2,style+'o',color=color,label=f'{view}, split {rep+1}')
            ax[1].plot(g.bank_A_stars,g.disagreement_log_sd_pct,style+'o',color=color)
            ax[2].plot(g.bank_A_stars+g.bank_B_stars,g.ensemble_cv_pct,style+'o',color=color)
    ax[0].set(xlabel='Training TICs per independent bank',ylabel='Mean of the two population-median CVs (%)',title=f'{group.title()}: increasing calibration data')
    ax[0].legend(fontsize=8)
    ax[1].set(xlabel='Training TICs per bank',ylabel='Median within-star SD of log(A/B), percent',title='Prediction disagreement, not a floor')
    ax[2].set(xlabel='Total TICs across disjoint A+B banks',ylabel='Population-median corrected CV (%)',title='Geometric-mean bank ensemble')
    for axis in ax:axis.set_xscale('log');axis.grid(alpha=.15)
    fig.savefig(OUT/f'bank_learning_{group}.png',dpi=170);plt.close(fig)


def feature_analysis(group='discovery'):
    roots={family:OUT/'full'/f'{family}_r3_l255_i200' for family in ['base','profiles','metadata','combined','static_transfer','gaia_references']}
    if not all((root/'complete.json').exists() for root in roots.values()):return
    frames={family:pd.read_parquet(root/'predictions.parquet') for family,root in roots.items()}
    base=frames['base'];records=[]
    for family,f in frames.items():
        pd.testing.assert_frame_equal(base[['source_row','tic_id','sector']],f[['source_row','tic_id','sector']])
        for view in ['all','qualified']:
            mask=base.evaluation_group.eq(group).to_numpy()
            if view=='qualified':mask &= base.systematic_score.gt(.95).to_numpy()
            control=score(base,base.log_correction_r3,mask);candidate=score(f,f.log_correction_r3,mask)
            records.append(dict(family=family,view=view,group=group,observations=int(mask.sum()),
                                cv_pct=float(candidate.median()),baseline_cv_pct=float(control.median()),
                                **paired_interval(control,candidate)))
    write_json(OUT/f'feature_metrics_{group}.json',records)
    if group=='discovery' and not (OUT/'feature_selection.json').exists():
        candidates=[r for r in records if r['view']=='qualified']
        selected=min(candidates,key=lambda r:r['cv_pct'])
        primary=min([r for r in candidates if r['family'] in ['base','profiles','metadata','combined']],key=lambda r:r['cv_pct'])
        write_json(OUT/'feature_selection.json',dict(selected_family=selected['family'],
            primary_four_family_selection=primary['family'],
            selected_utc=pd.Timestamp.now(tz='UTC').isoformat(),criterion='Lowest qualified discovery median CV at fixed strength and capacity',
            result=selected,discovery_metrics_sha256=sha256(OUT/'feature_metrics_discovery.json'),
            confirmation_opened=False))


def main():
    p=argparse.ArgumentParser();p.add_argument('--group',choices=['discovery','confirmation'],default='discovery');a=p.parse_args()
    if a.group=='confirmation':assert (OUT/'feature_selection.json').exists()
    bank_analysis(a.group);feature_analysis(a.group)


if __name__=='__main__':main()
