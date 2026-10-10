import glob,re,json
from pathlib import Path
import numpy as np,pandas as pd
import lightkurve as lk
files=glob.glob(str(Path.home()/'.lightkurve/cache/mastDownload/TESS/**/*_tp.fits'),recursive=True)
rows=[]
for p in files:
 try:
  t=lk.read(p); tic=int(t.targetid); f=np.asarray(t.flux.value,float); med=np.nanmedian(f,axis=0)
  mask=np.asarray(t.pipeline_mask,bool)
  bg=np.nanmedian(np.asarray(t.flux_bkg.value,float),axis=0) if getattr(t,'flux_bkg',None) is not None else np.zeros_like(med)
  # finite median image, background-subtracted where possible
  img=med-np.nanmedian(bg)
  img=np.where(np.isfinite(img),img,0)
  pos=np.argwhere(mask)
  w=np.clip(img,0,None)
  yy,xx=np.indices(img.shape)
  tot=float(np.nansum(np.clip(img,0,None)))
  ap=float(np.nansum(np.clip(img[mask],0,None)))
  if pos.size:
   cy=float(np.nansum(yy[mask]*np.clip(img[mask],0,None))/max(ap,1e-12)); cx=float(np.nansum(xx[mask]*np.clip(img[mask],0,None))/max(ap,1e-12))
   edge=float(min(cx-pos[:,1].min(),pos[:,1].max()-cx,cy-pos[:,0].min(),pos[:,0].max()-cy))
  else: cx=cy=edge=np.nan
  # concentration and off-mask bright-pixel flux proxy
  top=np.sort(np.clip(img[~mask],0,None).ravel())[-min(5,(~mask).sum()):]
  rows.append(dict(tic_id=tic,sector=int(t.sector),mask_area=int(mask.sum()),ap_capture=ap/max(tot,1e-12),offmask_flux=float(np.nansum(np.clip(img[~mask],0,None))/max(tot,1e-12)),offmask_top5=float(top.sum()/max(tot,1e-12)),mask_edge=float(edge),centroid_x=cx,centroid_y=cy,cam=int(t.camera),ccd=int(t.ccd)))
 except Exception as e: print('fail',p,e)
out=pd.DataFrame(rows)
lab=pd.read_parquet('training_data_v3.parquet',columns=['tic_id','flux_offset_loo','sector'])
lab=lab.groupby(['tic_id','sector']).flux_offset_loo.median().rename('loo').reset_index()
out=out.merge(lab,on=['tic_id','sector'],how='left')
star=out.groupby('tic_id').agg({**{c:'first' for c in out.columns if c not in ['tic_id','sector','loo']},'loo':'median'}).reset_index()
Path('eval/tpf_pilot_results').mkdir(exist_ok=True)
out.to_csv('eval/tpf_pilot_results/sector_features.csv',index=False); star.to_csv('eval/tpf_pilot_results/star_features.csv',index=False)
summary={'n_tpf':len(out),'n_stars':int(star.tic_id.nunique()),'n_labeled':int(star.loo.notna().sum())}
for c in ['mask_area','ap_capture','offmask_flux','offmask_top5','mask_edge']:
 z=star.dropna(subset=[c,'loo'])
 summary[c+'_r_loo']=float(z[c].corr(z.loo)) if len(z)>3 else None
print(json.dumps(summary,indent=2)); Path('eval/tpf_pilot_results/summary.json').write_text(json.dumps(summary,indent=2))
