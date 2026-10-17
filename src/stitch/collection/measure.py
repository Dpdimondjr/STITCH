"""Exact SAP/PDCSAP quality masks and cadence metadata measurements."""
import hashlib,io,warnings
import numpy as np
from astropy.io import fits
BITMASK=17087
DEFAULT_BITMASK=17087

def finite_median(v):
    return float(np.median(v)) if len(v) else None


def measure(body, rec):
    digest=hashlib.sha256(body).hexdigest()
    expected=rec.get('expected_raw_sha256')
    if isinstance(expected,str) and expected and digest!=expected:
        raise ValueError('Archive body hash differs from frozen original product')
    with fits.open(io.BytesIO(body),memmap=False) as h:
        if int(h[0].header['TICID'])!=int(rec['tic_id']) or int(h[0].header['SECTOR'])!=int(rec['sector']):
            raise ValueError('FITS TIC/sector identity mismatch')
        d=h[1].data;hdr=h[1].header
        t=np.asarray(d['TIME'],float);q=np.asarray(d['QUALITY'],np.int64)
        sap=np.asarray(d['SAP_FLUX'],float);pdc=np.asarray(d['PDCSAP_FLUX'],float)
        crowd=float(hdr.get('CROWDSAP',np.nan));capture=float(hdr.get('FLFRCSAP',np.nan))
        valid_header=np.isfinite(crowd) and 0<crowd<=1 and np.isfinite(capture) and 0<capture<=1
        out=dict(raw_sha256=digest,raw_bytes=len(body),n_cadences=len(d),
                 actual_cam=int(h[0].header.get('CAMERA',-1)),actual_ccd=int(h[0].header.get('CCD',-1)),
                 procver=str(h[0].header.get('PROCVER','')),data_rel=int(h[0].header.get('DATA_REL',-1)),
                 crowdsap=crowd if np.isfinite(crowd) else None,flfrcsap=capture if np.isfinite(capture) else None,
                 adjustment_header_valid=bool(valid_header),expected_hash_verified=isinstance(expected,str) and bool(expected))
        for name,quality in [('q0',q==0),('default',(q&BITMASK)==0)]:
            base=quality&np.isfinite(t)
            sm=base&np.isfinite(sap)&(sap>0);pm=base&np.isfinite(pdc)&(pdc>0)
            adj=(sap-(1-crowd)*np.median(sap[sm]))/capture if valid_header and sm.any() else np.full(len(sap),np.nan)
            am=sm&np.isfinite(adj)&(adj>0);common=am&pm
            for label,values,mask in [('sap',sap,sm),('pdcsap',pdc,pm),('adjusted_sap',adj,am),
                                      ('sap_common',sap,common),('pdcsap_common',pdc,common),('adjusted_sap_common',adj,common)]:
                out[label+'_median_'+name]=finite_median(values[mask])
                out['n_'+label+'_'+name]=int(mask.sum())
            out['start_'+name]=float(t[sm].min()) if sm.any() else None
            out['end_'+name]=float(t[sm].max()) if sm.any() else None
            for col,mask in [('SAP_FLUX_ERR',sm),('PDCSAP_FLUX_ERR',pm)]:
                v=np.asarray(d[col],float);ok=mask&np.isfinite(v)&(v>0)
                out[col.lower()+'_median_'+name]=finite_median(v[ok])
        out['status']='available' if out['n_sap_default']>0 or out['n_sap_q0']>0 else 'no_valid_sap'
        return out
def summarize(data,rec,origin):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        with fits.open(io.BytesIO(data),memmap=False) as h:
            assert int(h[0].header['TICID'])==rec['tic_id'] and int(h[0].header['SECTOR'])==rec['sector']
            for name,i in [('primary',0),('lightcurve',1),('aperture_header',2)]:
                if h[i].header.tostring()!=rec[name]:raise ValueError('LC header differs from frozen metadata')
            bits=np.frombuffer(rec['aperture_bytes'],dtype=rec['aperture_dtype']).reshape(int(rec['aperture_height']),int(rec['aperture_width']))
            np.testing.assert_array_equal(h[2].data,bits)
            d=h[1].data;q=np.asarray(d['QUALITY'],np.int64);t=np.asarray(d['TIME'],float)
            b=np.asarray(d['SAP_BKG'],float);pc1=np.asarray(d['POS_CORR1'],float);pc2=np.asarray(d['POS_CORR2'],float)
            out=dict(cadence_available=True,n_cadences=len(q),quality_zero_fraction=float(np.mean(q==0)),
                default_quality_fraction=float(np.mean((q&DEFAULT_BITMASK)==0)),
                scatter_flag_frac=float(np.mean((q&4096)!=0)),
                full_file_sha256=hashlib.sha256(data).hexdigest(),cadence_source=origin,
                procver=str(h[0].header.get('PROCVER','')),data_rel=int(h[0].header.get('DATA_REL',-1)),
                transferred_bytes=0 if origin.startswith('tess_cache/') else len(data))
            for label,mask in [('allfinite',np.ones(len(q),bool)),('default',(q&DEFAULT_BITMASK)==0),('q0',q==0)]:
                good=mask&np.isfinite(b);out['n_background_'+label]=int(good.sum())
                for name,value in [('median_sap_bkg',np.median),('p90_sap_bkg',lambda x:np.quantile(x,.9)),('bkg_rms',np.std)]:
                    out[name+'_'+label]=float(value(b[good])) if good.any() else np.nan
                for name,values in [('pos_corr1',pc1),('pos_corr2',pc2)]:
                    goodp=mask&np.isfinite(values);v=values[goodp]
                    out[name+'_median_'+label]=float(np.median(v)) if len(v) else np.nan
                    out[name+'_rms_'+label]=float(np.std(v)) if len(v) else np.nan
                    out[name+'_p05_'+label]=float(np.quantile(v,.05)) if len(v) else np.nan
                    out[name+'_p95_'+label]=float(np.quantile(v,.95)) if len(v) else np.nan
                out['jitter_rms_'+label]=float(np.hypot(out['pos_corr1_rms_'+label],out['pos_corr2_rms_'+label]))
            pdc=np.asarray(d['PDCSAP_FLUX'],float)
            valid=np.isfinite(pdc)&np.isfinite(t)&((q&DEFAULT_BITMASK)==0)
            out['lineage_pdcsap_median_default']=float(np.median(pdc[valid])) if valid.any() else np.nan
            out['lineage_n_pdcsap_default']=int(valid.sum())
            # Formal errors remain measurement diagnostics, never a total DC floor.
            for col in ['SAP_FLUX_ERR','PDCSAP_FLUX_ERR','SAP_BKG_ERR']:
                v=np.asarray(d[col],float);good=np.isfinite(v)&(v>0)&(q==0)
                out[col.lower()+'_median_q0']=float(np.median(v[good])) if good.any() else np.nan
            # Flux-weighted centroids are diagnostics, distinct from ensemble
            # pointing POS_CORR and from the incompletely documented legacy perr.
            for col in ['MOM_CENTR1','MOM_CENTR2','MOM_CENTR1_ERR','MOM_CENTR2_ERR']:
                v=np.asarray(d[col],float)
                for label,mask in [('allfinite',np.ones(len(q),bool)),('default',(q&DEFAULT_BITMASK)==0),('q0',q==0)]:
                    good=mask&np.isfinite(v)
                    out[col.lower()+'_median_'+label]=float(np.median(v[good])) if good.any() else np.nan
                    out[col.lower()+'_mean_'+label]=float(np.mean(v[good])) if good.any() else np.nan
                    if not col.endswith('_ERR'):
                        out[col.lower()+'_rms_'+label]=float(np.std(v[good])) if good.any() else np.nan
            good=(q==0)&np.isfinite(t)&np.isfinite(b)
            if good.sum()>1:
                tt=t[good];bb=b[good];first=tt<np.median(tt)
                out['background_half_shift']=float(np.median(bb[~first])-np.median(bb[first])) if first.any() and (~first).any() else np.nan
            else:out['background_half_shift']=np.nan
            return out
