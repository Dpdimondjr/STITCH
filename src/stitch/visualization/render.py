"""Render the existing card grid and two full-cadence panels from explicit inputs."""
import base64,gzip,json,math
from pathlib import Path
import numpy as np
import pandas as pd
from ..io import sha256


def summary(stars):
    result={}
    for ref in ['raw','baseline']:
        eligible=[s for s in stars if s['sc_after'] is not None]
        well=[s for s in eligible if s['n']>=5]
        harmed=[s for s in eligible if s['sc_after']>s['sc_'+ref]]
        harmed_well=[s for s in well if s['sc_after']>s['sc_'+ref]]
        finite=[s['improv_'+ref] for s in well if s['improv_'+ref] is not None]
        result[ref]=dict(total_stars=len(stars),stars_with_data=sum(s['n']>0 for s in stars),total_records=sum(s['n'] for s in stars),metric_stars=len(eligible),n5_stars=len(well),median_improv=float(np.median(finite)) if finite else None,harm_rate_n5=100*len(harmed_well)/len(well) if well else None,n_harmed=len(harmed),n_harmed_n5=len(harmed_well),n_cvz=sum(s['n']>=30 for s in stars))
    return result


def finite(value):return float(value) if np.isfinite(value) else None


def render(predictions,curves,out,model_spec=None):
    out=Path(out)
    if out.exists():raise FileExistsError('Use a new HTML output path')
    p=pd.read_parquet(predictions).sort_values(['tic_id','sector']);c=pd.read_parquet(curves)
    required=['tic_id','sector','correction','tmag','cam','ccd','pred_knn','systematic_score','has_gaia_anchor']
    if set(required)-set(p):raise ValueError(f'Prediction table requires {required}')
    if p.empty or p.duplicated(['tic_id','sector']).any():raise ValueError('Require unique nonempty TIC/sector predictions')
    if not np.isfinite(p[['correction','pred_knn','tmag','cam','ccd']].to_numpy()).all() or p[['correction','pred_knn']].le(0).any().any():raise ValueError('Invalid factors or display metadata')
    if set(['tic_id','sector','time','raw','quality'])-set(c):raise ValueError('Curves require tic_id, sector, time, raw, quality')
    c=c[np.isfinite(c.time)&np.isfinite(c.raw)&c.raw.gt(0)&c.quality.eq(0)].copy()
    if c.duplicated(['tic_id','sector','time']).any():raise ValueError('Duplicate retained cadence keys')
    medians=c.groupby(['tic_id','sector']).raw.median()
    p=p.set_index(['tic_id','sector']);p['raw_median']=medians.reindex(p.index)
    if p.raw_median.isna().any():raise ValueError('Every displayed sector needs retained QUALITY=0 cadences')
    p=p.reset_index();p['corrected_median']=p.raw_median/p.correction;p['baseline_median']=p.raw_median/p.pred_knn
    views={};packed={};coverage=[]
    for view in ['all','qualified']:
        stars=[]
        for tic,original in p.groupby('tic_id',sort=True):
            g=original if view=='all' else original[original.systematic_score.gt(.95)]
            vals=[g[name].to_numpy() for name in ['raw_median','baseline_median','corrected_median']]
            cvs=[float(100*np.std(v)/np.mean(v)) if len(v)>=2 else None for v in vals]
            norm=[(v/v.mean()).tolist() if len(v) else [] for v in vals]
            common=original.groupby(['cam','ccd']).size().sort_values(ascending=False).index[0]
            stars.append(dict(tic_id=int(tic),tmag=float(original.tmag.iloc[0]),n=len(g),cam=int(common[0]),ccd=int(common[1]),cameras=sorted(g.cam.astype(int).unique().tolist()),sc_raw=cvs[0],sc_baseline=cvs[1],sc_after=cvs[2],improv_raw=100*(1-cvs[2]/cvs[0]) if cvs[0] else None,improv_baseline=100*(1-cvs[2]/cvs[1]) if cvs[1] else None,sectors=g.sector.astype(int).tolist(),reference_raw_meds=norm[0],reference_baseline_meds=norm[1],cor_meds=norm[2]))
            coverage.append(dict(tic_id=int(tic),view=view,sectors=len(g),lost_sectors=len(original)-len(g),missing_anchors=int((~g.has_gaia_anchor).sum()),unknown_scores=int(g.systematic_score.isna().sum())))
        views[view]=dict(stars=stars,summary=summary(stars))
    for tic,g in p.groupby('tic_id',sort=True):
        star=dict(tic=int(tic),norms=[float(g.raw_median.mean()),float(g.corrected_median.mean())],sectors=[]);binaries=[]
        for row in g.to_dict('records'):
            z=c[c.tic_id.eq(tic)&c.sector.eq(row['sector'])].sort_values('time')
            binaries.extend([z.time.to_numpy('<f8').tobytes(),z.raw.to_numpy('<f4').tobytes(),z.quality.to_numpy('<u4').tobytes()])
            total=int(row.get('total_cadences',len(z)))
            star['sectors'].append(dict(sector=int(row['sector']),score=finite(row['systematic_score']),qualified=bool(row['systematic_score']>.95),camera=int(row['cam']),ccd=int(row['ccd']),factor=float(row['correction']),medians=[float(row['raw_median']),float(row['corrected_median'])],retained=len(z),total=total,rejected=total-len(z),missing_anchor=not bool(row['has_gaia_anchor']),references=int(row.get('ref_count',row.get('reference_count',0))),strap_overlap=finite(row.get('overlap_0',np.nan)),uri=row.get('data_uri',''),sha256=row.get('raw_sha256','')))
        binary=b''.join(binaries);star['packed']=base64.b64encode(gzip.compress(binary,compresslevel=6,mtime=0)).decode()
        if gzip.decompress(base64.b64decode(star['packed']))!=binary:raise RuntimeError('Cadence encoding failed')
        packed[str(int(tic))]=star
    stars=views['all']['stars'];scatter=[s[k] for s in stars for k in ['sc_raw','sc_baseline'] if s[k] is not None];improvements=[s[k] for s in stars for k in ['improv_raw','improv_baseline'] if s[k] is not None]
    m=dict(n_max=max(2,max(s['n'] for s in stars)),tmag_min=math.floor(p.tmag.min()),tmag_max=max(math.floor(p.tmag.min())+1,math.ceil(p.tmag.max())),sc_max=max(1,math.ceil(max(scatter,default=1))),improv_min=math.floor(min(improvements,default=-100)))
    spec=json.loads(Path(model_spec).read_text()) if model_spec else dict(model='Supplied corrections',target='See prediction provenance')
    payload=dict(model='STITCH supplied corrections',views=views,source_sha256=sha256(predictions))
    html=Path(__file__).with_name('dashboard.html').read_text()
    for name,value in [('DATA',payload),('M',m),('CADENCE_PACK',packed),('CADENCE_MODEL',spec)]:
        html=html.replace('__'+name+'__',json.dumps(value,separators=(',',':'),allow_nan=False).replace('<','\\u003c'))
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(html)
    pd.DataFrame(coverage).to_csv(out.with_suffix('.coverage.csv'),index=False)
    return dict(stars=len(packed),sectors=len(p),retained_cadences=len(c),html_sha256=sha256(out))
