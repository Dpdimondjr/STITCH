"""Catalog, aperture and PRF geometry with explicit local grid assets."""
import os
from pathlib import Path
from glob import glob
from functools import lru_cache
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.time import Time
import astropy.units as u

@lru_cache(maxsize=32)
def grid(localdatadir,cam,ccd):
    paths=[p for p in glob(os.path.join(localdatadir,f'cam{cam}_ccd{ccd}','*.fits')) if 'phot' not in p]
    return paths,np.array([int(p[-9:-5]) for p in paths]),np.array([int(p[-17:-13]) for p in paths])


@lru_cache(maxsize=128)
def tile(path):
    with fits.open(path,memmap=False) as h:return h[0].data.copy()


class CachedPRF:
    """Identical arithmetic and vendor locate(), caching only immutable FITS grids."""
    def __init__(self,cam,ccd,sector,colnum,rownum,localdatadir=None):
        self.cam,self.ccd,self.sector,self.colnum,self.rownum=cam,ccd,sector,colnum,rownum
        self.prfnsamp=9
        paths,cols,rows=grid(localdatadir,int(cam),int(ccd))
        subsets=[np.where(mask)[0] for mask in [(rows<rownum)&(cols<colnum),(rows>rownum)&(cols<colnum),
                                               (rows<rownum)&(cols>colnum),(rows>rownum)&(cols>colnum)]]
        if any(len(s)==0 for s in subsets):
            raise ValueError(f'Incomplete PRF grid for camera {cam}, CCD {ccd} at {localdatadir}; provide the required MAST grid assets')
        dist=np.sqrt((rows-rownum)**2.+(cols-colnum)**2.)
        inds=[s[np.argmin(dist[s])] for s in subsets]
        points=sorted([(cols[i],rows[i],tile(paths[i])) for i in inds])
        (x1,y1,q11),(_x1,y2,q12),(x2,_y1,q21),(_x2,_y2,q22)=points
        self.prf=(q11*(x2-colnum)*(y2-rownum)+q21*(colnum-x1)*(y2-rownum)+
                  q12*(x2-colnum)*(rownum-y1)+q22*(colnum-x1)*(rownum-y1))/((x2-x1)*(y2-y1)+0.)
        reshaped=np.zeros((11,11,13,13))
        for i in range(9):
            for j in range(9):reshaped[i+1,j+1,:,:]=self.prf[8-i::9,8-j::9]
        for j in range(1,10):
            reshaped[j,0,:,:]=np.append(reshaped[j,-2,:,1:],np.zeros((13,1)),axis=1)
            reshaped[j,-1,:,:]=np.append(np.zeros((13,1)),reshaped[j,1,:,:-1],axis=1)
        for i in range(11):
            reshaped[0,i,:,:]=np.append(reshaped[-2,i,1:,:],np.zeros((1,13)),axis=0)
            reshaped[-1,i,:,:]=np.append(np.zeros((1,13)),reshaped[1,i,:-1,:],axis=0)
        self.reshaped=reshaped


def header_features(h):
    h0,h1,h2=h;out={}
    names={'CROWDSAP':'header_crowdsap','FLFRCSAP':'header_flfrcsap','PDCVAR':'header_pdcvar',
           'PR_WGHT2':'header_pr_wght2','PDC_NOI':'header_pdc_noi','PDC_CORP':'header_pdc_corp',
           'PDC_TOTP':'header_pdc_totp','CDPP0_5':'header_cdpp0_5','CDPP1_0':'header_cdpp1_0',
           'CDPP2_0':'header_cdpp2_0','PDC_TOT':'header_pdc_tot','PDC_COR':'header_pdc_cor'}
    for key,name in names.items():
        try:out[name]=float(h1[key])
        except (TypeError,ValueError,KeyError):out[name]=np.nan
    try:out['header_teff']=float(h0['TEFF'])
    except (TypeError,ValueError,KeyError):out['header_teff']=np.nan
    out['pipeline_method']=str(h1.get('PDCMETHD',''))
    out['procver']=str(h0.get('PROCVER',''));out['data_rel']=int(h0.get('DATA_REL',-1))
    return out
def positions(table,tic,header,midtime):
    def num(c):return np.asarray(np.ma.filled(table[c],np.nan),float)
    ids=np.asarray(table['ID']).astype(str);flags=np.asarray(np.ma.filled(table['POSflag'],''),str)
    native=np.isin(flags,['gaia2','tmgaia2'])&np.isfinite(num('RA_orig'))&np.isfinite(num('Dec_orig'))
    ra=np.where(native,num('RA_orig'),num('ra'));dec=np.where(native,num('Dec_orig'),num('dec'))
    sky=SkyCoord(ra=ra*u.deg,dec=dec*u.deg,pm_ra_cosdec=np.nan_to_num(num('pmRA'))*u.mas/u.yr,
        pm_dec=np.nan_to_num(num('pmDEC'))*u.mas/u.yr,obstime=Time(np.where(native,2015.5,2000.),format='jyear'))
    moved=sky.apply_space_motion(Time(midtime+2457000,format='jd',scale='tdb'))
    x,y=WCS(header).celestial.all_world2pix(moved.ra.deg,moved.dec.deg,0)
    target=ids==str(tic);assert target.sum()==1
    disposition=np.asarray(np.ma.filled(table['disposition'],''),str);mag=num('Tmag')
    usable=(~np.isin(disposition,['DUPLICATE','ARTIFACT'])&np.isfinite(mag)&np.isfinite(x)&np.isfinite(y))|target
    return pd.DataFrame(dict(tic_id=ids,x=x,y=y,tmag=mag,gaiarp=num('gaiarp'),is_target=target,usable=usable,native_gaia=native))
def masses(prf,xy,shape,mask,strap,chunk=256):
    """Integrate the vendor's normalized 13x13 locate template without padding arrays."""
    pixel_samples=np.arange(-1/18,19.1/18,1/9);output=[];pad=20
    for start in range(0,len(xy),chunk):
        p=np.asarray(xy[start:start+chunk],float)+pad
        ii=np.floor(p+.5).astype(int);phase=(p+.5)%1
        above=np.searchsorted(pixel_samples,phase,side='left');below=above-1
        cb,rb=below.T;ca,ra=above.T;cf,rf=phase.T
        x1=pixel_samples[cb];x2=pixel_samples[ca];y1=pixel_samples[rb];y2=pixel_samples[ra]
        def v(z):return z[:,None,None]
        q11=prf.reshaped[rb,cb];q12=prf.reshaped[ra,cb];q21=prf.reshaped[rb,ca];q22=prf.reshaped[ra,ca]
        z=(q11*v(x2-cf)*v(y2-rf)+q21*v(cf-x1)*v(y2-rf)+
           q12*v(x2-cf)*v(rf-y1)+q22*v(cf-x1)*v(rf-y1))/(v(x2-x1)*v(y2-y1))
        z/=z.sum(axis=(1,2))[:,None,None]
        xx=(ii[:,0]-pad)[:,None,None]+np.arange(-6,7)[None,None,:]
        yy=(ii[:,1]-pad)[:,None,None]+np.arange(-6,7)[None,:,None]
        valid=(xx>=0)&(xx<shape[1])&(yy>=0)&(yy<shape[0])
        ap=mask[np.clip(yy,0,shape[0]-1),np.clip(xx,0,shape[1]-1)]&valid
        st=strap[np.clip(yy,0,shape[0]-1),np.clip(xx,0,shape[1]-1)]&ap
        output.append(np.column_stack([(z*valid).sum(axis=(1,2)),(z*ap).sum(axis=(1,2)),(z*st).sum(axis=(1,2))]))
    return np.concatenate(output)


def feature(primary,lc,header,bits,tic,sector,table,columns,midtime=None, *, prf_dir):
    assert int(primary['TICID'])==tic and int(primary['SECTOR'])==sector
    cam=int(primary['CAMERA']);ccd=int(primary['CCD'])
    nominal_c=float(lc['CROWDSAP']);nominal_q=float(lc['FLFRCSAP'])
    if nominal_c<=0 or nominal_q<=0:raise ValueError('Nonpositive header crowding/capture')
    midpoint=(float(lc['TSTART'])+float(lc['TSTOP']))/2
    if midtime is None:midtime=midpoint
    mask=((bits.astype(int)&2)!=0)&((bits.astype(int)&1)!=0);shape=bits.shape
    if not mask.any():raise ValueError('No aperture pixels')
    sky=WCS(header).celestial;physical=WCS(header,key='P')
    x,y=sky.all_world2pix(header['RA_OBJ'],header['DEC_OBJ'],0);col,row=physical.all_pix2world(x,y,0)
    yy,xx=np.indices(shape);pcol,_=physical.all_pix2world(xx,yy,0);strap=np.isin(np.rint(pcol-44).astype(int),columns)
    prf=CachedPRF(cam,ccd,sector,float(np.clip(col,45.001,2091.999)),float(np.clip(row,1.001,2047.999)),
        localdatadir=str(Path(prf_dir)/('start_s0001' if sector<4 else 'start_s0004')))
    cat=positions(table,tic,header,midtime);target=cat[cat.is_target].iloc[0]
    near=cat.usable&cat.x.between(-6,shape[1]+5)&cat.y.between(-6,shape[0]+5)
    cat=cat[near|cat.is_target].copy();mass=masses(prf,cat[['x','y']].to_numpy(),shape,mask,strap)
    cat['stamp_capture']=mass[:,0];cat['aperture_capture']=mass[:,1];cat['strap_capture']=mass[:,2]
    target_strap=float(cat.loc[cat.is_target,'strap_capture'].iloc[0])
    cat['relative_flux']=10**(-.4*(cat.tmag-target.tmag))
    cat['expected_stamp_contribution']=cat.stamp_capture*cat.relative_flux
    cat['expected_aperture_contribution']=cat.aperture_capture*cat.relative_flux
    cat=cat.sort_values(['is_target','expected_stamp_contribution'],ascending=[False,False])
    xy=cat[['x','y']].to_numpy(float);target_mask=cat.is_target.to_numpy();stamp=cat.expected_stamp_contribution.to_numpy()
    kept=[];rejected=[]
    for i in range(len(cat)):
        if len(kept)>=18:
            rejected.extend(range(i,len(cat)));break
        if not target_mask[i] and stamp[i]<.001:rejected.append(i);continue
        if kept and np.min(np.linalg.norm(xy[kept]-xy[i],axis=1))<.15:rejected.append(i);continue
        kept.append(i)
    selected=cat.iloc[kept];assert selected.iloc[0].is_target
    expected=selected.expected_aperture_contribution.to_numpy();capture=float(selected.iloc[0].aperture_capture)
    crowd=float(expected[0]/expected.sum())
    if not np.isfinite(capture+crowd) or min(capture,crowd)<=0:raise ValueError('Invalid scene capture/crowding')
    neighbor=selected[~selected.is_target]
    return dict(actual_cam=cam,actual_ccd=ccd,actual_col=float(col),actual_row=float(row),
        midtime=float(midtime),header_midpoint=midpoint,capture_term=float(np.log(capture/nominal_q)),
        crowding_term=float(np.log(nominal_c/crowd)),prf_capture=capture,catalog_scene_crowding=crowd,
        nominal_capture=nominal_q,nominal_crowding=nominal_c,overlap_0=float(target_strap/capture),
        neighbor_aperture_ratio=float(neighbor.expected_aperture_contribution.sum()/capture),
        neighbor_strap_ratio=float((neighbor.strap_capture*neighbor.relative_flux).sum()/capture),
        area_fraction=float((mask&strap).sum()/mask.sum()),aperture_pixels=int(mask.sum()),sources=len(kept),
        omitted_expected_aperture_flux_relative_target=float(cat.iloc[rejected].expected_aperture_contribution.sum()/expected[0]) if rejected else 0.,
        target_native_gaia=bool(target.native_gaia),target_header_position_difference=float(np.hypot(target.x-x,target.y-y)),
        procver=str(primary.get('PROCVER','')),data_rel=int(primary.get('DATA_REL',-1)),
        prf_grid_clamp_pixels=float(np.hypot(np.clip(col,45.001,2091.999)-col,np.clip(row,1.001,2047.999)-row)))
