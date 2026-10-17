import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from stitch.features.context import FEATURES
from stitch.reference.prepare import prepare
from stitch.hgb.train import residual,train
from stitch.reference.safe import hgb_features


@pytest.fixture
def inputs(tmp_path):
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
    return tmp_path


def test_anchor_is_not_centered_away():
    f=pd.DataFrame(dict(tic_id=[1,1,2,2],sector_median=[100.,101.,200.,202.],pred_knn=[1.]*4,gaia_log_anchor=[4.,4.,5.,5.],has_gaia_anchor=[True]*4))
    old=residual(f,np.zeros(4));f.gaia_log_anchor+=.01
    np.testing.assert_allclose(residual(f,np.zeros(4))-old,-.01,atol=1e-12)
    assert abs(np.median(old[:2]))>.1


def test_source_alias_exclusion():
    ref=pd.DataFrame(dict(tic_id=[1],physical_group_id=['same'],systematic_score=[.99]))
    query=pd.DataFrame(dict(tic_id=[2],physical_group_id=['same']))
    with pytest.raises(ValueError,match='physical_group_id'):hgb_features(ref,query)


def test_prepare_and_five_round_fit(inputs):
    a=inputs;out=a/'prepared'
    f=prepare(a/'metadata.parquet',a/'labels.parquet',a/'splits.parquet',out)
    labels=pd.read_parquet(a/'labels.parquet');labels.loc[labels.tic_id.ge(100100),'sector_median']*=1.5;labels.to_parquet(a/'changed_labels.parquet',index=False)
    g=prepare(a/'metadata.parquet',a/'changed_labels.parquet',a/'splits.parquet',a/'changed_prepared')
    from stitch.features.context import REFCOLS
    pd.testing.assert_frame_equal(f[REFCOLS+['gaia_log_anchor']],g[REFCOLS+['gaia_log_anchor']])
    config=a/'small.json';config.write_text(json.dumps(dict(max_iter=2,max_leaf_nodes=7,min_samples_leaf=4)))
    train(out,a/'model',a/'identity.parquet',config)
    from stitch.hgb.model import FrozenPhysicalModel
    model=FrozenPhysicalModel(a/'model');value=model.predict(f[f.role.ne('train')])
    assert np.isfinite(value).all() and (value>0).all()
    assert len(list((a/'model/physical').glob('round_*.pkl')))==5
    with pytest.raises(FileExistsError):train(out,a/'model',a/'identity.parquet',config)


def test_cnf_gaia_training(inputs):
    torch=pytest.importorskip('torch');pytest.importorskip('zuko');torch.set_num_threads(1)
    from stitch.cnf.train import train as train_cnf
    from stitch.cnf.model import CNFModel
    a=inputs;f=prepare(a/'metadata.parquet',a/'labels.parquet',a/'splits.parquet',a/'cnf_prepared','cnf')
    train_cnf(a/'cnf_prepared',a/'cnf_model',a/'identity.parquet',epochs=1,batch_size=128)
    assert json.loads((a/'cnf_model/model.json').read_text())['target']=='gaia_rp'
    result=CNFModel(a/'cnf_model').predict(f[f.role.ne('train')])
    assert np.isfinite(result).all() and (result>0).all()


def test_matched_cnf_uses_hgb_prior_and_source_exclusions(inputs):
    import shutil
    torch=pytest.importorskip('torch');zuko=pytest.importorskip('zuko');torch.set_num_threads(1)
    from stitch.cnf.model import CNFModel
    from stitch.bundles import seal_bundle
    a=inputs;prepared=a/'physical_prepared'
    f=prepare(a/'metadata.parquet',a/'labels.parquet',a/'splits.parquet',prepared,'hgb')
    root=a/'matched_cnf';root.mkdir()
    for name in ['reference_bank.parquet','cohort_manifest.parquet',*[f'color_fold_{i}.pkl' for i in range(5)]]:
        shutil.copy2(prepared/name,root/name)
    config=dict(features=1,context=len(FEATURES),transforms=1,hidden_features=[8],bins=4)
    flow=zuko.flows.NSF(**config)
    for parameter in flow.parameters():torch.nn.init.zeros_(parameter)
    zeros=pd.Series(0.,index=FEATURES);ones=pd.Series(1.,index=FEATURES)
    torch.save(dict(state=flow.state_dict(),config=config,feature_cols=FEATURES,
        medians=zeros,means=zeros,stds=ones,y_median=0.,y_std=1.),root/'model.pt')
    (root/'model.json').write_text(json.dumps(dict(target='gaia_rp',reference_family='hgb',beta=1.,clip=None)))
    seal_bundle(root,dict(kind='cnf'))
    model=CNFModel(root)
    q=f.iloc[:2].copy();q['pred_knn']=[1.,2.];q['knn20_median']=3.
    result=model.predict(q)
    np.testing.assert_allclose(result[1]/result[0],2.,rtol=1e-7)
    q['pred_knn']=0.
    with pytest.raises(ValueError,match='reference correction'):model.predict(q)
    with pytest.raises(ValueError,match='historical cohort'):model.attach_references(f[f.tic_id.eq(f.tic_id.iloc[0])])
