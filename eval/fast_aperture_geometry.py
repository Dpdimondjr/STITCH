"""Vectorized PRF mass integrals, preserving the frozen source-selection rule."""
import numpy as np
import pandas as pd
from astropy.wcs import WCS
from extract_geometry_training_transfer import positions
from fit_scene_investigation import PRF
from pathlib import Path


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


def feature(primary,lc,header,bits,tic,sector,table,columns,midtime=None):
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
    prf=PRF.TESS_PRF(cam,ccd,sector,float(np.clip(col,45.001,2091.999)),float(np.clip(row,1.001,2047.999)),
        localdatadir=str(Path('tess_cache/prf_models')/('start_s0001' if sector<4 else 'start_s0004')))
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
