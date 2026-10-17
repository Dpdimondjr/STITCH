"""Synthetic plumbing example; never use these numbers as science validation."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from stitch.features.context import FEATURES

def make_demo(tmp_path):
    rng=np.random.default_rng(41);records=[]
    for star in range(140):
        rp=8.5+(star%30)/10;color=.2+(star%25)/10
        for sector in [10,11,12]:
            row={name:0. for name in FEATURES}
            row.update(source_row=len(records),tic_id=100000+star,sector=sector,cam=1,ccd=1,actual_cam=1,col=100+(star%35)*40,row=100+(star//35)*300,tmag=rp+.2,gaiarp=rp,bp_rp=color,crowdsap=.99,focal_plane_temp=20.,focal_plane_temp_std=.1,n_sectors_obs=3,pdcvar=1.,jitter_rms=.01)
            row['sector_median']=10000*10**(-.4*(rp-10))*(1+.002*(sector-11)+rng.normal(0,.0002))
            records.append(row)
    f=pd.DataFrame(records);s=f[['source_row','tic_id','sector']].copy();s['physical_group_id']=s.tic_id.astype(str)
    s['role']=np.where(s.tic_id<100100,'train',np.where(s.tic_id<100130,'validation','test'));s['systematic_score']=.99
    f.drop(columns='sector_median').to_parquet(tmp_path/'metadata.parquet',index=False)
    f[['source_row','tic_id','sector','sector_median']].to_parquet(tmp_path/'labels.parquet',index=False)
    s.to_parquet(tmp_path/'splits.parquet',index=False)
    identity=pd.DataFrame(dict(ID=np.arange(100000,100140),GAIA=np.arange(100000,100140).astype(str),ra=np.arange(140,dtype=float),dec=0.));identity.to_parquet(tmp_path/'identity.parquet',index=False)
    curves=[]
    for row in f.itertuples():
        phase=np.arange(32)*2*np.pi/32
        curves.append(pd.DataFrame(dict(tic_id=row.tic_id,sector=row.sector,time=27*row.sector+np.arange(32)/2,raw=row.sector_median*(1+.001*np.sin(phase)),quality=0)))
    pd.concat(curves,ignore_index=True).to_parquet(tmp_path/'curves.parquet',index=False)
    return tmp_path

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("out",type=Path);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);print(make_demo(a.out))
