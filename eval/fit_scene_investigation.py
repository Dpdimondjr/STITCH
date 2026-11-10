"""Independent per-sector target/neighbor amplitudes using catalog positions.

Nominal PRFs are a hypothesis, not photometric truth. The target is never held
constant across sectors. Every fit retains full diagnostics, including failures.
"""
import os
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/stitch_mpl')
import argparse
from pathlib import Path
import sys
import warnings
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy.wcs import WCS
import astropy.units as u
from scipy.optimize import nnls, minimize
from scipy.ndimage import gaussian_filter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from investigate_residual_atlas import save_json, log

ROOT=Path('eval/residual_investigation_20261011/scene')
CACHE=Path('eval/repaired_aperture_pilot_20261010/stage_2_extraction')
sys.path.insert(0,str(Path('tess_cache/prf_vendor').resolve()))
import PRF


def project_fit(image, templates, background, weight, valid):
    """Profile unconstrained background, then NNLS source amplitudes.

    Source templates are NOT renormalized within the stamp: amplitudes represent
    total source flux, retaining aperture/cutout losses.
    """
    y=image[valid]*weight[valid]
    A=templates[:,valid].T*weight[valid,None]
    B=background[:,valid].T*weight[valid,None]
    q,_=np.linalg.qr(B,mode='reduced')
    ap=A-q@(q.T@A);yp=y-q@(q.T@y)
    scale=np.maximum(np.linalg.norm(ap,axis=0),1e-30)
    amp,_=nnls(ap/scale,yp,maxiter=100*len(templates))
    amp/=scale
    bg=np.linalg.lstsq(B,y-A@amp,rcond=None)[0]
    model=np.einsum('i,ijk->jk',amp,templates)+np.einsum('i,ijk->jk',bg,background)
    residual=image-model
    condition=np.linalg.cond(ap/scale)
    return amp,bg,model,float(np.mean((residual[valid]*weight[valid])**2)),condition


def catalog_positions(path,tic,header,midtime):
    t=Table.read(path)
    def numeric(c):return np.asarray(np.ma.filled(t[c],np.nan),float)
    ids=np.asarray(t['ID']).astype(str)
    ra=numeric('ra');dec=numeric('dec');mag=numeric('Tmag')
    pmra=numeric('pmRA');pmdec=numeric('pmDEC')
    # TIC v8 RA/Dec are propagated to epoch J2000.0 (MAST release notes).
    sky=SkyCoord(ra=ra*u.deg,dec=dec*u.deg,
        pm_ra_cosdec=np.nan_to_num(pmra)*u.mas/u.yr,
        pm_dec=np.nan_to_num(pmdec)*u.mas/u.yr,obstime=Time(2000.,format='jyear'))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        propagated=sky.apply_space_motion(new_obstime=Time(midtime+2457000,format='jd',scale='tdb'))
    x,y=WCS(header).celestial.all_world2pix(propagated.ra.deg,propagated.dec.deg,0)
    dispositions=np.asarray(np.ma.filled(t['disposition'],''),str)
    usable=~np.isin(dispositions,['DUPLICATE','ARTIFACT'])&np.isfinite(mag)&np.isfinite(x)&np.isfinite(y)
    target=ids==str(tic);usable|=target
    out=pd.DataFrame(dict(tic_id=ids,x=x,y=y,tmag=mag,proper_motion_available=np.isfinite(pmra)&np.isfinite(pmdec),usable=usable,is_target=target))
    assert target.sum()==1
    return out


def templates_at(prf,positions,shape,shift=(0.,0.)):
    # Padding makes off-stamp sources safe even when their centers lie beyond
    # the 13-pixel support handled by PRF.locate's direct slicing.
    pad=20;size=(shape[0]+2*pad,shape[1]+2*pad)
    return np.array([prf.locate(x+shift[0]+pad,y+shift[1]+pad,size)[pad:-pad,pad:-pad]
                     for x,y in positions])


def optimize_scene(image,prf,positions,background,weight,valid,start=(0.,0.)):
    def objective(shift):
        templates=templates_at(prf,positions,image.shape,shift)
        try:return project_fit(image,templates,background,weight,valid)[3]
        except (RuntimeError,np.linalg.LinAlgError):return 1e100
    opt=minimize(objective,np.asarray(start),method='Powell',bounds=[(-.5,.5),(-.5,.5)],
        options={'maxiter':20,'xtol':.004,'ftol':1e-5})
    templates=templates_at(prf,positions,image.shape,opt.x)
    fitted=project_fit(image,templates,background,weight,valid)
    return opt.x,templates,fitted,bool(opt.success)


def fit_observation(r):
    curves=np.load(CACHE/f'tic_{r.tic_id}_s{r.sector:04d}.npz')
    image=curves['median_image'].astype(float);shape=image.shape;mask=curves['mask_pipeline']
    with fits.open(r.tpf_path,memmap=True) as h:
        header=h[2].header;bits=h[2].data
        sky=WCS(header).celestial
        x,y=sky.all_world2pix(header['RA_OBJ'],header['DEC_OBJ'],0)
        col,row=WCS(header,key='P').all_pix2world(x,y,0)
        epoch='start_s0001' if r.sector<4 else 'start_s0004'
        # The official grid ends at physical cols 45/2092, rows 1/2048.
        # Record boundary interpolation clamp; sources outside silicon aren't fit.
        gridcol=float(np.clip(col,45.001,2091.999));gridrow=float(np.clip(row,1.001,2047.999))
        prf=PRF.TESS_PRF(r.cam,r.ccd,r.sector,gridcol,gridrow,
            localdatadir=str(Path('tess_cache/prf_models')/epoch))
        tab=h[1].data
        ix=pd.Index(np.asarray(tab['CADENCENO'],np.int64)).get_indexer(curves['cadenceno'])
        if (ix<0).any():raise ValueError('Cadence alignment lost')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            pixel_error=np.nanmedian(np.asarray(tab['FLUX_ERR'][ix],float),axis=0)
    valid=((bits&1)!=0)&np.isfinite(image)&np.isfinite(pixel_error)&(pixel_error>0)
    weight=1/np.maximum(pixel_error,np.median(pixel_error[valid])*.25)
    cat=catalog_positions(ROOT/'catalogs'/f'tic_{r.tic_id}.ecsv',r.tic_id,header,np.median(curves['time']))
    target=cat[cat.is_target].iloc[0]
    near=cat.usable&cat.x.between(-6,shape[1]+5)&cat.y.between(-6,shape[0]+5)
    cat=cat[near|cat.is_target].copy()
    templates=templates_at(prf,cat[['x','y']].to_numpy(),shape)
    cat['stamp_capture']=templates.sum(axis=(1,2))
    cat['aperture_capture']=templates[:,mask].sum(axis=1)
    cat['catalog_relative_flux']=10**(-.4*(cat.tmag-target.tmag))
    cat['expected_stamp_contribution']=cat.stamp_capture*cat.catalog_relative_flux
    cat['expected_aperture_contribution']=cat.aperture_capture*cat.catalog_relative_flux
    cat=cat.sort_values(['is_target','expected_stamp_contribution'],ascending=[False,False])
    keep=[];rejected=[]
    for i,s in cat.iterrows():
        if not s.is_target and s.expected_stamp_contribution<.001:
            rejected.append(i);continue
        if keep and np.min(np.linalg.norm(cat.loc[keep,['x','y']].to_numpy()-s[['x','y']].to_numpy(float),axis=1))<.15:
            rejected.append(i);continue
        if len(keep)>=18:rejected.append(i);continue
        keep.append(i)
    selected=cat.loc[keep].copy()
    assert selected.iloc[0].is_target
    positions=selected[['x','y']].to_numpy(float)
    yy,xx=np.indices(shape);background=np.array([np.ones(shape),(xx-xx.mean())/shape[1],(yy-yy.mean())/shape[0]])
    shift,templates,fit,success=optimize_scene(image,prf,positions,background,weight,valid)
    amp,bg,model,objective,condition=fit
    source_model=np.einsum('i,ijk->jk',amp,templates)
    residual=image-model;apflux=np.sum(image[mask]);target_ap=amp[0]*templates[0,mask].sum()
    neighbor_ap=np.sum(source_model[mask])-target_ap
    fitted_crowd=target_ap/max(target_ap+neighbor_ap,1e-20)
    fitted_capture=templates[0,mask].sum()
    mismatch=np.log(max(fitted_capture,1e-15)/r.lc_flfrcsap)+np.log(r.lc_crowdsap/max(fitted_crowd,1e-15))
    # Positive mismatch predicts PDCSAP overestimating the target brightness
    # under the nominal crowding/capture-only correction, before PDC changes.
    ap_l1=float(np.sum(abs(residual[mask]))/max(abs(apflux),1e-20))
    omitted=float(cat.loc[rejected,'expected_aperture_contribution'].sum()/max(selected.iloc[0].aperture_capture,1e-20)) if rejected else 0.
    gate=bool(success and amp[0]>0 and condition<1e4 and ap_l1<.05 and omitted<.05 and max(abs(shift))<.49)
    record=dict(tic_id=r.tic_id,sector=r.sector,pair_id=r.pair_id,pilot_group=r.pilot_group,
        diagnostic_half=r.diagnostic_half,cam=r.cam,ccd=r.ccd,fit_success=success,quality_pass=gate,
        sources_fit=len(amp),target_flux=amp[0],target_capture=fitted_capture,fitted_crowdsap=fitted_crowd,
        pipeline_crowdsap=r.lc_crowdsap,pipeline_capture=r.lc_flfrcsap,
        predicted_log_mismatch=mismatch,shift_x=shift[0],shift_y=shift[1],condition=condition,
        aperture_l1_fraction=ap_l1,full_stamp_l1_fraction=float(np.sum(abs(residual[valid]))/np.sum(abs(image[valid]))),
        omitted_catalog_aperture_flux_relative_target=omitted,
        predicted_background_aperture_fraction=float(np.sum((model-source_model)[mask])/apflux),
        strongest_neighbor_flux_ratio=float(np.max(amp[1:])/max(amp[0],1e-20)) if len(amp)>1 else 0,
        target_catalog_header_position_delta_pixels=float(np.hypot(target.x-x,target.y-y)),
        prf_grid_clamp_pixels=float(np.hypot(gridcol-col,gridrow-row)),
        residual_pct=r.signed_residual_pct,stitch_flux=r.median_stitch_pdcsap,
        raw_pdcsap=r.median_pdcsap,sap_nominal=r.median_sap_nominal,
        median_sap=r.median_sap,nominal_pipeline_only_prediction=r.median_sap*r.lc_crowdsap/r.lc_flfrcsap)
    selected['fitted_flux']=amp;selected['tic_target']=r.tic_id;selected['sector']=r.sector
    # Injections are made into the observed image. Refit all amplitudes and
    # the astrometric nuisance shift; no across-sector target constraint.
    injections=[]
    for name,addition,expected in [
        ('target_plus_1pct',.01*amp[0]*templates[0],.01*amp[0]),
        ('target_minus_1pct',-.01*amp[0]*templates[0],-.01*amp[0]),
        ('target_shifted_shape_plus_1pct',.01*amp[0]*templates_at(prf,positions[:1],shape,shift+np.array([.10,0]))[0],.01*amp[0])]:
        _,_,refit,_=optimize_scene(image+addition,prf,positions,background,weight,valid,start=shift)
        recovered=refit[0][0]-amp[0]
        injections.append(dict(tic_id=r.tic_id,sector=r.sector,kind=name,expected_flux_change=expected,
            recovered_flux_change=recovered,recovery_fraction=recovered/expected if expected else np.nan,quality_pass=gate))
    if len(amp)>1:
        j=1+np.argmax(amp[1:]*templates[1:,mask].sum(axis=1))
        for name,temp in [('neighbor_plus_10pct',templates[j]),('neighbor_blurred_plus_10pct',gaussian_filter(templates[j],.25))]:
            _,_,refit,_=optimize_scene(image+.1*amp[j]*temp,prf,positions,background,weight,valid,start=shift)
            injections.append(dict(tic_id=r.tic_id,sector=r.sector,kind=name,expected_flux_change=0.,
                recovered_flux_change=refit[0][0]-amp[0],spurious_target_change_pct=100*(refit[0][0]/max(amp[0],1e-20)-1),quality_pass=gate))
    np.savez_compressed(ROOT/f'fit_tic_{r.tic_id}_s{r.sector:04d}.npz',image=image,model=model,residual=residual,
        target=amp[0]*templates[0],neighbors=source_model-amp[0]*templates[0],mask=mask,valid=valid,
        positions=positions,amplitudes=amp,shift=shift)
    return record,selected,pd.DataFrame(injections)


def association(frame,column):
    z=frame[['tic_id','pair_id','diagnostic_half','residual_pct',column]].dropna().copy()
    z['x']=z[column]-z.groupby('tic_id')[column].transform('mean')
    z['y']=z.residual_pct-z.groupby('tic_id').residual_pct.transform('mean')
    z['w']=1/z.groupby('tic_id').tic_id.transform('size')
    def corr(g):
        den=np.sqrt(np.sum(g.w*g.x*g.x)*np.sum(g.w*g.y*g.y))
        return np.sum(g.w*g.x*g.y)/den if den else np.nan
    return {'all':corr(z),'half_A':corr(z[z.diagnostic_half=='A']),'half_B':corr(z[z.diagnostic_half=='B']),
            'stars':z.tic_id.nunique(),'observations':len(z)}


def summarize(frame,injections):
    per=[]
    for tic,g in frame.groupby('tic_id'):
        if len(g)<3:continue
        for method in ['target_flux','stitch_flux','raw_pdcsap','sap_nominal']:
            f=g[method].to_numpy()
            per.append(dict(tic_id=tic,pilot_group=g.pilot_group.iloc[0],method=method,sectors=len(g),
                all_quality_pass=bool(g.quality_pass.all()),cv_pct=100*np.std(f)/np.mean(f) if np.mean(f)>0 else np.nan))
    per=pd.DataFrame(per);per.to_csv(ROOT/'per_star_scene_cv.csv',index=False)
    metric={}
    for selection,f in [('all_completed',frame),('quality_pass',frame[frame.quality_pass])]:
        metric[selection]={'predicted_log_mismatch':association(f,'predicted_log_mismatch'),
            'empirical_pdc_vs_scene':association(f.assign(ratio=np.log(f.raw_pdcsap/f.target_flux.clip(lower=1e-20))),'ratio')}
    inj={}
    for name,g in injections.groupby('kind'):
        col='spurious_target_change_pct' if name.startswith('neighbor') else 'recovery_fraction'
        inj[name]={'all_completed_n':len(g),'all_quantiles':g[col].quantile([.05,.5,.95]).to_dict(),
            'quality_pass_n':int(g.quality_pass.sum()),'quality_pass_quantiles':g.loc[g.quality_pass,col].quantile([.05,.5,.95]).to_dict()}
    result={'completed_observations':len(frame),'stars':frame.tic_id.nunique(),'quality_pass_observations':int(frame.quality_pass.sum()),
        'stars_all_four_quality_pass':int(frame.groupby('tic_id').quality_pass.all().sum()),
        'median_aperture_l1_fraction':frame.aperture_l1_fraction.median(),
        'median_full_stamp_l1_fraction':frame.full_stamp_l1_fraction.median(),
        'signed_associations':metric,'pixel_injections':inj,
        'limitations':['20 selected matched pairs; not a population or untouched holdout sample.',
            'Catalog positions propagated from J2000; one free common translation per sector, bounded to half a pixel.',
            'Nominal spatial PRF is shared within each small stamp; target and neighbor fluxes are independent every sector.',
            'Bright neighbors outside the stamp have poorly constrained amplitudes; catalog amplitudes are not forced.',
            'Residual aperture L1 <5%, condition <1e4, omitted expected aperture contamination <5%, non-boundary shift are fixed quality gates, not proof of 1% photometric accuracy.',
            'Empirical PDC/scene ratio shares the measured PDCSAP response and is not an independent feature.',
            'Exact-family injections test numerical recovery; shifted/broadened injections expose only limited PRF mismatch cases.',
            'No physical correction or new test CV claimed.']}
    save_json(ROOT/'results.json',result)
    fig,axes=plt.subplots(2,2,figsize=(11,8),constrained_layout=True)
    for half,c in [('A','#226b8c'),('B','#b34e33')]:
        g=frame[frame.diagnostic_half==half].copy()
        x=g.predicted_log_mismatch-g.groupby('tic_id').predicted_log_mismatch.transform('mean')
        y=g.residual_pct-g.groupby('tic_id').residual_pct.transform('mean')
        axes[0,0].scatter(100*x,y,s=18,alpha=.6,label=half,color=c)
    axes[0,0].set(xlabel='Centered predicted crowding/capture mismatch (%)',ylabel='STITCH residual (%)',title='Does the nominal scene model predict the signed error?');axes[0,0].legend(title='Pair half')
    axes[0,1].hist(100*frame.aperture_l1_fraction,bins=25,color='#457b9d');axes[0,1].axvline(5,color='#b34e33',ls='--');axes[0,1].set(xlabel='Aperture absolute fit residual / aperture flux (%)',ylabel='Observations',title='Model adequacy before claiming photometric truth')
    p=per.pivot(index='tic_id',columns='method',values='cv_pct')
    axes[1,0].scatter(p.stitch_flux,p.target_flux,s=24,alpha=.7)
    lim=max(p.stitch_flux.max(),p.target_flux.max());axes[1,0].plot([0,lim],[0,lim],color='.7',ls='--')
    axes[1,0].set(xlabel='STITCH CV on selected sectors (%)',ylabel='Independent scene-fit target CV (%)',title='Free sector amplitudes; lower CV is not sufficient')
    g=injections[injections.kind.str.startswith('target')]
    for name,z in g.groupby('kind'):
        sortedv=np.sort(z.recovery_fraction.clip(-1,3));axes[1,1].plot(sortedv,np.arange(1,len(z)+1)/len(z),label=name.replace('target_','').replace('_',' '))
    axes[1,1].axvline(1,color='.6',ls='--');axes[1,1].set(xlabel='Recovered / injected target change (display clipped −1..3)',ylabel='Fraction of fits',title='Injection preservation, including failed-quality fits');axes[1,1].legend(fontsize=7)
    fig.savefig(ROOT/'scene_fit_summary.png',dpi=180);plt.close(fig)
    # Show best and worst fits, with explicit deterministic selection.
    chosen=pd.concat([frame.sort_values('aperture_l1_fraction').head(2),frame.sort_values('aperture_l1_fraction').tail(2)])
    fig,axes=plt.subplots(4,4,figsize=(12,11),constrained_layout=True)
    for axrow,r in zip(axes,chosen.itertuples(index=False)):
        z=np.load(ROOT/f'fit_tic_{r.tic_id}_s{r.sector:04d}.npz')
        scale=max(np.nanpercentile(abs(z['image']),99),1)
        for ax,key in zip(axrow,['image','model','target','residual']):
            vals=np.arcsinh(z[key]/(scale*.02))
            im=ax.imshow(vals,origin='lower',cmap='RdBu_r' if key=='residual' else 'magma',vmin=-4 if key=='residual' else 0,vmax=4)
            ax.contour(z['mask'],levels=[.5],colors=['#3ee5a8'],linewidths=.7)
            ax.set(title=f'{key} • TIC {r.tic_id} S{r.sector}',xlabel='Stamp column',ylabel='Stamp row')
    fig.suptitle('Two smallest and two largest aperture-fit residuals; common asinh scaling within each row')
    fig.savefig(ROOT/'scene_fit_examples.png',dpi=180);plt.close(fig)
    (ROOT/'RESULTS.md').write_text('# Target and neighbor scene fit\n\n'+(ROOT/'results.json').read_text()+'\n\n![Summary](scene_fit_summary.png)\n\n![Examples](scene_fit_examples.png)\n')


def main():
    global ROOT
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--limit',type=int)
    parser.add_argument('--root',default=str(ROOT));args=parser.parse_args();ROOT=Path(args.root)
    sample=pd.read_parquet(ROOT/'sample.parquet')
    if args.limit:sample=sample.head(args.limit)
    records=[];sources=[];injections=[];errors=[]
    for i,r in enumerate(sample.itertuples(index=False)):
        try:
            path=ROOT/f'result_tic_{r.tic_id}_s{r.sector:04d}.parquet'
            if path.exists():
                record=pd.read_parquet(path).iloc[0].to_dict()
                source=pd.read_parquet(ROOT/f'sources_tic_{r.tic_id}_s{r.sector:04d}.parquet')
                inj=pd.read_parquet(ROOT/f'injections_tic_{r.tic_id}_s{r.sector:04d}.parquet')
            else:
                record,source,inj=fit_observation(r)
                pd.DataFrame([record]).to_parquet(path,index=False)
                source.to_parquet(ROOT/f'sources_tic_{r.tic_id}_s{r.sector:04d}.parquet',index=False)
                inj.to_parquet(ROOT/f'injections_tic_{r.tic_id}_s{r.sector:04d}.parquet',index=False)
            records.append(record);sources.append(source);injections.append(inj)
            log(f'{i+1}/{len(sample)} TIC {r.tic_id} S{r.sector}: ap residual {record["aperture_l1_fraction"]:.3f}; pass={record["quality_pass"]}')
        except Exception as exc:
            errors.append(dict(tic_id=r.tic_id,sector=r.sector,error=f'{type(exc).__name__}: {exc}'))
            log(f'{r.tic_id} S{r.sector}: {exc}')
    save_json(ROOT/'fit_errors.json',errors)
    if records:
        frame=pd.DataFrame(records);inj=pd.concat(injections,ignore_index=True)
        frame.to_parquet(ROOT/'fit_measurements.parquet',index=False);inj.to_parquet(ROOT/'pixel_injections.parquet',index=False)
        pd.concat(sources,ignore_index=True).to_parquet(ROOT/'source_measurements.parquet',index=False)
        summarize(frame,inj)
    if errors:raise RuntimeError(f'{len(errors)} scene fits failed; inspect fit_errors.json')


if __name__=='__main__':main()
