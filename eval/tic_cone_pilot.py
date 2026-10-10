import json,time
from pathlib import Path
import numpy as np,pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u
from astroquery.mast import Catalogs

rng=np.random.default_rng(20260929)
df=pd.read_parquet('training_data_v3.parquet', columns=['tic_id','sector','tmag','ra','dec','flux_offset_loo','sector_median'])
df=df[(df.tmag<=13)&df.ra.notna()&df.dec.notna()&df.flux_offset_loo.notna()]
star=df.groupby('tic_id').agg(ra=('ra','first'),dec=('dec','first'),tmag=('tmag','median'),loo=('flux_offset_loo','median'),abs_loo=('flux_offset_loo',lambda x: np.median(np.abs(x-1))),n=('sector','nunique')).reset_index()
star=star[star.n>=5]
sample=star.sample(min(100,len(star)),random_state=20260929)
rows=[]; failures=[]
for j,rec in enumerate(sample.itertuples(index=False),1):
 try:
  tab=Catalogs.query_region(SkyCoord(rec.ra,rec.dec,unit='deg'), radius=150*u.arcsec, catalog='Tic')
  names=set(str(c).lower() for c in tab.colnames)
  # TIC column naming varies; normalize
  def col(*xs):
   for x in xs:
    if x in tab.colnames:return np.asarray(tab[x])
    if x.lower() in names:
     return np.asarray(tab[[c for c in tab.colnames if c.lower()==x.lower()][0]])
   return None
  ids=col('ID','ticid'); tm=col('Tmag','TESSMAG');
  if ids is None or tm is None: raise RuntimeError('missing ID/Tmag')
  ids=np.asarray(ids); tm=np.asarray(tm,dtype=float)
  keep=np.isfinite(tm)&(tm<=17)&(ids.astype(str)!=str(rec.tic_id))
  tm=tm[keep]
  # TESS flux ratio using Tmag; cap to avoid overflow
  ratio=np.sum(10**(-0.4*(tm-rec.tmag))) if len(tm) else 0.0
  rows.append(dict(tic_id=int(rec.tic_id),n_neighbors=int(len(tm)),cont_ratio=float(ratio),bright_neighbors=int(np.sum(tm<=13)),loo=float(rec.loo),abs_loo=float(rec.abs_loo),tmag=float(rec.tmag),n_sectors=int(rec.n)))
 except Exception as e: failures.append((int(rec.tic_id),type(e).__name__,str(e)[:160]))
 print(j, len(rows), 'fail',len(failures), flush=True)
out=pd.DataFrame(rows)
Path('eval/tic_cone_pilot_results').mkdir(exist_ok=True)
out.to_csv('eval/tic_cone_pilot_results/sample.csv',index=False)
summary={'n_requested':len(sample),'n_success':len(out),'n_fail':len(failures),'failures':failures}
for c in ['n_neighbors','cont_ratio','bright_neighbors']:
 if len(out)>3:
  summary[c+'_r_loo']=float(out[c].corr(out.loo)); summary[c+'_r_abs_loo']=float(out[c].corr(out.abs_loo))
summary['feature_quantiles']={c:[float(x) for x in out[c].quantile([.01,.5,.99])] for c in ['n_neighbors','cont_ratio','bright_neighbors']} if len(out) else {}
Path('eval/tic_cone_pilot_results/summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
