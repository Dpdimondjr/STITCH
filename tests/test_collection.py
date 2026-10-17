import io,json,threading,time
import numpy as np
import pandas as pd
import pytest
from astropy.io import fits
from stitch.collection import collect as module


def body(tic):
    primary=fits.PrimaryHDU();primary.header['TICID']=tic;primary.header['SECTOR']=10
    primary.header['CAMERA']=1;primary.header['CCD']=1
    n=40;columns=[]
    values={'TIME':np.arange(n,dtype=float),'QUALITY':np.zeros(n,dtype=np.int32),
            'SAP_FLUX':np.linspace(1000,1001,n),'PDCSAP_FLUX':np.linspace(1100,1101,n),
            'SAP_FLUX_ERR':np.ones(n),'PDCSAP_FLUX_ERR':np.ones(n),
            'SAP_BKG':np.full(n,10.),'SAP_BKG_ERR':np.ones(n),
            'POS_CORR1':np.zeros(n),'POS_CORR2':np.zeros(n),
            'MOM_CENTR1':np.full(n,4.),'MOM_CENTR2':np.full(n,4.),
            'MOM_CENTR1_ERR':np.ones(n),'MOM_CENTR2_ERR':np.ones(n)}
    values['TIME'][0]=np.nan;values['QUALITY'][1]=4096
    for name,value in values.items():columns.append(fits.Column(name=name,array=value,format='J' if name=='QUALITY' else 'D'))
    lc=fits.BinTableHDU.from_columns(columns);lc.header['CROWDSAP']=.9;lc.header['FLFRCSAP']=.8
    aperture=fits.ImageHDU(np.full((3,3),3,dtype=np.int32));stream=io.BytesIO();fits.HDUList([primary,lc,aperture]).writeto(stream)
    return stream.getvalue()


def test_bounded_commit_resume_and_hashes(tmp_path,monkeypatch):
    requests=tmp_path/'requests.parquet';pd.DataFrame([dict(tic_id=i,sector=10,data_uri=f'https://example.invalid/{i}') for i in range(1,5)]).to_parquet(requests,index=False)
    lock=threading.Lock();state={'active':0,'peak':0,'calls':0}
    def load(rec):
        with lock:
            state['active']+=1;state['calls']+=1;state['peak']=max(state['peak'],state['active'])
        time.sleep(.03);data=body(rec['tic_id'])
        with lock:state['active']-=1
        return data,'test_memory',len(data)
    monkeypatch.setattr(module,'load_body',load)
    out=tmp_path/'collected';module.collect(requests,out)
    assert state['peak']==2 and state['calls']==4
    assert not list(out.rglob('*.fits'))
    measurement=pd.read_parquet(out/'measurements/0.parquet').iloc[0]
    assert measurement.n_sap_q0==38 and measurement.n_pdcsap_q0==38
    module.collect(requests,out,resume=True);assert state['calls']==4
    with pytest.raises(ValueError,match='one or two'):module.collect(requests,tmp_path/'too_many',workers=3)
    with (out/'cadences/0.parquet').open('ab') as stream:stream.write(b'changed')
    with pytest.raises(ValueError,match='Resume output changed'):module.collect(requests,out,resume=True)
