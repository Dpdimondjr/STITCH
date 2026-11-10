"""One prespecified CNF feature ablation, with calibration-only checkpoint choice."""
import os
os.environ['OMP_NUM_THREADS']='4';os.environ['OPENBLAS_NUM_THREADS']='1'
import json,shutil,hashlib
from pathlib import Path
import numpy as np,pandas as pd,joblib,torch,zuko
import train_matched_gaia_cnf as original

ROOT=Path('eval/restricted_gaia_cnf_20261017')
SOURCE=Path('eval/matched_gaia_cnf_20261017')
REMOVED=['cdpp1_0','pdcvar','pdc_noi','pr_wght2','pdc_corp','pdc_totp','pdctotp_x_crowdsap','dev_pdcvar_loo',
 'mom_centr1_rms_default','mom_centr2_rms_default','dev_mom_centr1_rms_default','dev_mom_centr2_rms_default',
 'perr_col','perr_row','quality_zero_fraction','default_quality_fraction','n_cadences','scatter_flag_frac']


class Model:
    def __init__(self,root=ROOT):
        self.root=Path(root);s=json.loads((self.root/'selection.json').read_text())
        assert original.sha(self.root/s['selected'])==s['checkpoint_sha256']
        self.c=torch.load(self.root/s['selected'],weights_only=False);self.prep=self.c['preprocessing']
        self.flow=zuko.flows.NSF(**self.c['config']);self.flow.load_state_dict(self.c['state']);self.flow.eval()
    def predict(self,f):
        p=self.prep;cols=p['features']
        x=((f[cols].replace([np.inf,-np.inf],np.nan).fillna(p['medians'])-p['means'])/p['stds']).to_numpy(np.float32)
        _,m=original.distribution_outputs(self.flow,x)
        return np.exp(np.log(f.pred_knn.to_numpy(float))+m*p['y_std']+p['y_median'])


def main():
    assert not ROOT.exists();assert shutil.disk_usage('.').free>4*2**30
    ROOT.mkdir();(ROOT/'source_snapshot').mkdir()
    prep=joblib.load(SOURCE/'preprocessing.pkl');columns=[c for c in prep['features'] if c not in REMOVED]
    assert len(columns)==102
    paths=[SOURCE/n for n in ['train_x.npy','train_y.npy','calibration_x.npy','calibration_y.npy','training_rows.parquet','calibration_rows.parquet','preprocessing.pkl','protocol.json']]
    config=dict(original.CONFIG,context=len(columns))
    protocol=dict(created_utc=pd.Timestamp.now(tz='UTC').isoformat(),hypothesis='Quiet-star CNF input dependencies may transfer poorly to variables. Remove the previously defined 18-input HGB ablation, without selecting inputs from binary outcomes.',
        model='Four NSF transforms, 128/128 hidden, 12 bins; same seed/optimizer/batch/early stopping as original full CNF.',
        seed=original.SEED,features=columns,removed=REMOVED,flow_config=config,
        training='Exact 457123 rows and 36507 stars from full CNF; same standardized target and surviving feature preprocessing.',
        target='log(sector_median / pred_knn)-gaia_log_anchor; exact source-safe training-only color transfer; no star centering.',
        selection='Only minimum qualified calibration median-star CV, <=60 epochs, 12-epoch patience. No binary/confirmation/test-based checkpoint or feature selection.',
        science='Apply selected checkpoint to all original binary sources using the frozen signal-aware protocol, including failed endpoints and controls. This is development after inspection of full CNF; no pristine replication claim.',
        limits='Aperture/background and catalog geometry can still respond indirectly to variability. Restricted does not certify instrumental-only causality.',
        input_sha256={str(p):original.sha(p) for p in paths},downloads=0,production_changed=False)
    original.save(ROOT/'protocol.json',protocol)
    for p in [Path(__file__),Path(original.__file__),Path('research_defaults.json')]:shutil.copy2(p,ROOT/'source_snapshot'/p.name)
    idx=[prep['features'].index(c) for c in columns]
    for name in ['train_x','calibration_x']:
        a=np.load(SOURCE/(name+'.npy'),mmap_mode='r');np.save(ROOT/(name+'.npy'),np.asarray(a[:,idx],order='C'))
    for name in ['train_y.npy','calibration_y.npy','training_rows.parquet','calibration_rows.parquet']:
        (ROOT/name).symlink_to((SOURCE/name).resolve())
    prep['features']=columns
    for key in ['medians','means','stds']:prep[key]=prep[key].reindex(columns)
    joblib.dump(prep,ROOT/'preprocessing.pkl')
    original.ROOT=ROOT;original.CONFIG=config
    original.train()
    model=Model();q=pd.read_parquet(SOURCE/'predictions.parquet');q=q[q.evaluation_group.isin(['calibration','confirmation'])].copy()
    f=pd.read_parquet('eval/full_rebuild_20261012/features.parquet').set_index('source_row').loc[q.source_row].reset_index()
    q['log_correction_restricted_cnf']=np.log(model.predict(f))
    q.to_parquet(ROOT/'predictions.parquet',index=False)
    rows=[]
    for cohort,g in q.groupby('evaluation_group'):
        for view,z in [('all',g),('qualified',g[g.systematic_score.gt(.95)])]:
            for m in ['raw','hgb','cnf','restricted_cnf']:
                cv=original.cv_values(z,z['log_correction_'+m].to_numpy())
                rows.append(dict(cohort=cohort,view=view,model=m,rows=len(z),stars=len(cv),median_cv_pct=float(cv.median()),missing_anchors=int((~z.has_gaia_anchor).sum())))
    pd.DataFrame(rows).to_csv(ROOT/'quiet_metrics.csv',index=False)
    for path,h in protocol['input_sha256'].items():assert original.sha(path)==h
    original.save(ROOT/'verification.json',dict(status='passed',input_hashes_unchanged=True,training_rows=457123,features=102,
        selection_uses_only_calibration=True,gaia_target_preserved=True,no_downloads=True,production_changed=False))
    print(pd.DataFrame(rows).to_string(index=False),flush=True)


if __name__=='__main__':main()
