"""NSF training; Gaia RP by default, relative targets only by explicit request."""
import copy,json,shutil
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import zuko
from torch.utils.data import DataLoader,TensorDataset
from ..hgb.train import residual
from ..io import write_json
from ..bundles import seal_bundle

FEATURES=['col','row','delta_sub_col','delta_sub_row','sector','cam','ccd','tmag','crowdsap','cdpp1_0','pdcvar','jitter_rms','pdc_noi','pr_wght2','gaiarp','focal_plane_temp','knn20_median','knn20_mad','knn20_radius','reference_count']


def train(prepared,out,identity_catalog,epochs=60,target='gaia_rp',batch_size=2048):
    prepared=Path(prepared);out=Path(out)
    if out.exists():raise FileExistsError('Use a new model directory')
    if epochs<1 or batch_size<1:raise ValueError('Positive epochs and batch size required')
    if json.loads((prepared/'preparation.json').read_text())['family']!='cnf':raise ValueError('Expected CNF preparation')
    f=pd.read_parquet(prepared/'features.parquet').merge(pd.read_parquet(prepared/'labels.parquet'),on='source_row',validate='one_to_one')
    tr=f[f.fit_row&~f.unsupported_reference];va=f[f.evaluation_group.eq('calibration')&~f.unsupported_reference]
    if tr.empty or va.empty:raise ValueError('Supported training and calibration rows required')
    med=tr[FEATURES].replace([np.inf,-np.inf],np.nan).median().fillna(0.)
    clean=tr[FEATURES].replace([np.inf,-np.inf],np.nan).fillna(med);means=clean.mean();stds=clean.std().replace(0,1).fillna(1.)
    x=((clean-means)/stds).to_numpy(np.float32)
    vx=((va[FEATURES].replace([np.inf,-np.inf],np.nan).fillna(med)-means)/stds).to_numpy(np.float32)
    raw=residual(tr,np.zeros(len(tr)),target,prior='knn20_median');ym=float(np.median(raw));ys=float(np.std(raw))
    if not np.isfinite(ys) or ys<=0:raise ValueError('Degenerate training target')
    y=((raw-ym)/ys).astype(np.float32)
    vy=((residual(va,np.zeros(len(va)),target,prior='knn20_median')-ym)/ys).astype(np.float32)
    seed=20260916;torch.manual_seed(seed);np.random.seed(seed)
    config=dict(features=1,context=len(FEATURES),transforms=4,hidden_features=[128,128],bins=12)
    flow=zuko.flows.NSF(**config);opt=torch.optim.Adam(flow.parameters(),lr=3e-4,weight_decay=1e-5)
    sched=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,patience=5,factor=.5)
    loader=DataLoader(TensorDataset(torch.from_numpy(x),torch.from_numpy(y[:,None])),batch_size=batch_size,shuffle=True,generator=torch.Generator().manual_seed(seed))
    best=float('inf');state=None;stale=0;history=[]
    for epoch in range(epochs):
        flow.train()
        for xx,yy in loader:
            loss=-flow(xx).log_prob(yy).mean();opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(flow.parameters(),5);opt.step()
        flow.eval();value=0.
        with torch.no_grad():
            for start in range(0,len(vx),8192):value-=float(flow(torch.from_numpy(vx[start:start+8192])).log_prob(torch.from_numpy(vy[start:start+8192,None])).sum())
        value/=len(vx);sched.step(value);history.append(dict(epoch=epoch+1,calibration_nll=value))
        if value<best-1e-4:best=value;state=copy.deepcopy(flow.state_dict());stale=0
        else:stale+=1
        print(f'CNF epoch {epoch+1}: calibration NLL {value:.6f}',flush=True)
        if stale>=12:break
    if state is None:raise ValueError('Training failed to produce a finite calibration loss')
    out.mkdir(parents=True)
    torch.save(dict(state=state,config=config,feature_cols=FEATURES,medians=med,means=means,stds=stds,y_median=ym,y_std=ys),out/'model.pt')
    write_json(out/'model.json',dict(target=target,beta=1.,clip=None,estimator='conditional median',history=history,missing_training_anchors=int((~tr.has_gaia_anchor).sum()),new_gaia_cnf=target=='gaia_rp',historical_checkpoint=False,training_rows=len(tr),calibration_rows=len(va),unsupported_training_rows=int(f.loc[f.fit_row,'unsupported_reference'].sum())))
    for p in prepared.glob('color_fold_*.pkl'):shutil.copy2(p,out/p.name)
    for name in ['reference_bank.parquet','cohort_manifest.parquet','preparation.json']:shutil.copy2(prepared/name,out/name)
    shutil.copy2(identity_catalog,out/'identity_catalog.parquet')
    f[['sector','cam','ccd','focal_plane_temp','focal_plane_temp_std']].drop_duplicates().to_parquet(out/'thermal.parquet',index=False)
    seal_bundle(out,dict(kind='cnf',target=target,status='new research fit; not automatically promoted'))
