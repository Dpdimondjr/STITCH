"""Conditional-median NSF inference, preserving archived scaling and bounds."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import zuko
from ..bundles import verify_bundle
from ..reference.safe import cnf_features
from ..hgb.model import FrozenPhysicalModel, validate_reference_bank


class CNFModel:
    def __init__(self,root):
        self.root=Path(root);verify_bundle(self.root)
        self.spec=json.loads((self.root/'model.json').read_text())
        self.reference_family=self.spec.get('reference_family','cnf')
        if self.reference_family not in ['cnf','hgb']:
            raise ValueError('Unknown CNF reference feature family')
        self.prior_column='pred_knn' if self.reference_family=='hgb' else 'knn20_median'
        self.reference_count_column='ref_count' if self.reference_family=='hgb' else 'reference_count'
        self.c=torch.load(self.root/'model.pt',map_location='cpu',weights_only=False)
        self.flow=zuko.flows.NSF(**self.c['config'])
        self.flow.load_state_dict(self.c['state']);self.flow.eval()
        if self.reference_family=='hgb':
            self.bank=pd.read_parquet(self.root/'reference_bank.parquet')
            self.cohort=pd.read_parquet(self.root/'cohort_manifest.parquet')
            validate_reference_bank(self.bank,self.cohort)

    def attach_references(self,frame):
        if self.reference_family=='hgb':
            if frame.tic_id.nunique()!=1:
                raise ValueError('Attach matched CNF references one target at a time')
            return FrozenPhysicalModel.attach_references(self,frame,int(frame.tic_id.iloc[0]))
        bank=pd.read_parquet(self.root/'reference_bank.parquet')
        if set(frame.physical_group_id.astype(str))&set(bank.physical_group_id.astype(str)):
            raise ValueError('Query overlaps the frozen reference source groups')
        q=frame.reset_index(drop=True).copy();spatial=cnf_features(bank,q)
        for col in spatial:q[col]=spatial[col].to_numpy()
        q['unsupported_reference']=q.knn20_median.isna();q['knn20_median']=q.knn20_median.fillna(1.)
        return q

    def predict(self,frame,batch_size=4096):
        c=self.c;cols=c['feature_cols']
        if set(cols+[self.prior_column])-set(frame):raise ValueError('Missing prepared CNF features')
        prior=(frame[self.prior_column].to_numpy(float) if self.reference_family=='hgb'
               else frame[self.prior_column].to_numpy())
        if not np.all(np.isfinite(prior)&(prior>0)):raise ValueError('Invalid CNF reference correction')
        x=((frame[cols].replace([np.inf,-np.inf],np.nan).fillna(c['medians'])-c['means'])/c['stds']).to_numpy(np.float32)
        if not np.isfinite(x).all():raise ValueError('Invalid preprocessed CNF context')
        result=[]
        with torch.no_grad():
            for start in range(0,len(x),batch_size):
                d=self.flow(torch.from_numpy(x[start:start+batch_size]))
                count=len(x[start:start+batch_size])
                shape=(count,1) if self.reference_family=='hgb' else (1,count,1)
                y=d.transform.inv(torch.zeros(shape)).squeeze().numpy().reshape(-1)
                if self.reference_family=='hgb':
                    if self.spec['beta']!=1 or self.spec.get('clip') is not None:
                        raise ValueError('Matched Gaia CNF requires beta=1 and no clipping')
                    value=np.exp(np.log(prior[start:start+batch_size])+y*c['y_std']+c['y_median'])
                else:
                    value=prior[start:start+batch_size]*np.exp(self.spec['beta']*(y*c['y_std']+c['y_median']))
                if self.spec.get('clip') is not None:value=np.clip(value,*self.spec['clip'])
                result.append(value)
        result=np.concatenate(result)
        if not np.all(np.isfinite(result)&(result>0)):raise ValueError('Invalid CNF correction')
        return result
