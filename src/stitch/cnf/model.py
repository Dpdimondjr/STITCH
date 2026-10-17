"""Conditional-median NSF inference, preserving archived scaling and bounds."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import zuko
from ..bundles import verify_bundle
from ..reference.safe import cnf_features


class CNFModel:
    def __init__(self,root):
        self.root=Path(root);verify_bundle(self.root)
        self.spec=json.loads((self.root/'model.json').read_text())
        self.c=torch.load(self.root/'model.pt',map_location='cpu',weights_only=False)
        self.flow=zuko.flows.NSF(**self.c['config'])
        self.flow.load_state_dict(self.c['state']);self.flow.eval()

    def attach_references(self,frame):
        bank=pd.read_parquet(self.root/'reference_bank.parquet')
        if set(frame.physical_group_id.astype(str))&set(bank.physical_group_id.astype(str)):
            raise ValueError('Query overlaps the frozen reference source groups')
        q=frame.reset_index(drop=True).copy();spatial=cnf_features(bank,q)
        for col in spatial:q[col]=spatial[col].to_numpy()
        q['unsupported_reference']=q.knn20_median.isna();q['knn20_median']=q.knn20_median.fillna(1.)
        return q

    def predict(self,frame,batch_size=4096):
        c=self.c;cols=c['feature_cols']
        if set(cols+['knn20_median'])-set(frame):raise ValueError('Missing prepared CNF features')
        x=((frame[cols].replace([np.inf,-np.inf],np.nan).fillna(c['medians'])-c['means'])/c['stds']).to_numpy(np.float32)
        if not np.isfinite(x).all():raise ValueError('Invalid preprocessed CNF context')
        result=[]
        with torch.no_grad():
            for start in range(0,len(x),batch_size):
                d=self.flow(torch.from_numpy(x[start:start+batch_size]))
                y=d.transform.inv(torch.zeros((1,len(x[start:start+batch_size]),1))).squeeze().numpy().reshape(-1)
                value=frame.knn20_median.to_numpy()[start:start+batch_size]*np.exp(self.spec['beta']*(y*c['y_std']+c['y_median']))
                if self.spec.get('clip') is not None:value=np.clip(value,*self.spec['clip'])
                result.append(value)
        return np.concatenate(result)
