"""Calibration-only selection first, then quiet/science development evaluation."""
from matched_sap_hgb import *
from validate_baseline_boundaries import jump,pseudogaps

def quiet_metrics(group):
    rows=[];individual=[]
    for branch in BRANCHES:
        p=pd.read_parquet(ROOT/branch/'predictions.parquet');p=p[p.evaluation_group.eq(group)]
        for view in ['all','qualified']:
            z=p if view=='all' else p[p.systematic_score.gt(.95)]
            for method in ['sector_median','corrected']:
                cv=z.groupby('tic_id')[method].agg(lambda x:100*np.std(x,ddof=1)/np.mean(x))
                e=np.log(z[method])-z.gaia_log_anchor
                rows.append(dict(branch=branch,group=group,view=view,method=method,stars=len(cv),sectors=len(z),median_cv_pct=float(cv.median()),anchor_log_rmse=float(np.sqrt(np.mean(e**2))),reference_fallbacks=int(z.ref_count.eq(0).sum())))
                individual.extend(dict(branch=branch,group=group,view=view,method=method,tic_id=int(t),cv_pct=float(v)) for t,v in cv.items())
    return rows,individual

def select():
    assert not (ROOT/'selection.json').exists()
    rows,individual=quiet_metrics('calibration')
    eligible=[r for r in rows if r['view']=='qualified' and r['method']=='corrected' and r['branch']!='pdcsap']
    save(ROOT/'selection.json',dict(selected_sap=min(eligible,key=lambda r:r['median_cv_pct'])['branch'],calibration_candidates=eligible,confirmation_outcomes_opened=False,science_outcomes_opened=False,evaluator_sha256=sha(__file__),created_utc=pd.Timestamp.now(tz='UTC').isoformat()))
    pd.DataFrame(rows).to_csv(ROOT/'calibration_scores.csv',index=False)
    pd.DataFrame(individual).to_csv(ROOT/'calibration_per_star.csv',index=False)
    print(json.dumps(json.loads((ROOT/'selection.json').read_text()),indent=2),flush=True)

def quiet():
    assert (ROOT/'selection.json').exists()
    rows,individual=quiet_metrics('confirmation')
    pd.concat([pd.read_csv(ROOT/'calibration_scores.csv'),pd.DataFrame(rows)]).to_csv(ROOT/'quiet_scores.csv',index=False)
    pd.concat([pd.read_csv(ROOT/'calibration_per_star.csv'),pd.DataFrame(individual)]).to_csv(ROOT/'quiet_per_star.csv',index=False)
    print(pd.DataFrame(rows).to_string(index=False),flush=True)

def scientific_predict(branch,f):
    root=ROOT/branch;bank=pd.read_parquet(root/'reference_bank.parquet')
    cfg=json.loads((ROOT/'freeze.json').read_text());cols=cfg['features']
    assert not set(bank.physical_group_id.astype(str))&set(f.physical_group_id.astype(str))
    assert not set(bank.tic_id)&set(f.tic_id)
    parts=[]
    for fold in sorted(f.reference_fold.unique()):
        q=f[f.reference_fold.eq(fold)].drop(columns=REF+['gaia_log_anchor'],errors='ignore').copy()
        ref=bank[bank.reference_fold.ne(fold)]
        rf=reference_features(ref,q);rf.index=q.index;q=q.join(rf)
        color=joblib.load(root/f'color_fold_{fold}.pkl')
        q['gaia_log_anchor']=-.4*np.log(10)*(q.gaiarp-10)+color.predict(color_design(q.bp_rp))
        parts.append(q)
    q=pd.concat(parts).sort_values('sector');impute=joblib.load(root/'imputation.pkl')
    x=q[cols].astype(float).replace([np.inf,-np.inf],np.nan).fillna(impute).to_numpy(np.float32)
    q['correction']=q.pred_knn*np.exp(sum(joblib.load(root/f'round_{i}.pkl').predict(x) for i in range(5)))
    return q

def science():
    assert (ROOT/'selection.json').exists()
    source=pd.read_parquet(ROOT/'source_selection.parquet');metadata=pd.read_parquet(VARIABLE/'predictions.parquet')
    support=[];factors=[];daily=[];attr=[];edges=[]
    for tic,pp in metadata.groupby('tic_id'):
        f=pd.read_parquet(VARIABLE/'inference'/str(tic)/'features.parquet')
        mag=float(f.tmag.iloc[0]);color=float(f.bp_rp.iloc[0])
        near=source[(source.tmag-mag).abs().lt(.75)&(source.bp_rp-color).abs().lt(.35)]
        nt=int(near.evaluation_group.eq('train').sum());nv=int(near.evaluation_group.eq('calibration').sum())
        eligible=nt>=3 and nv>=1 and f.has_gaia_anchor.all() and np.isfinite(color)
        support.append(dict(tic_id=int(tic),tmag=mag,bp_rp=color,near_training_stars=nt,near_calibration_stars=nv,eligible=bool(eligible),sectors=len(f),qualified_sectors=int(pp.qualified.sum()),color_transfer_outside_0_3p3=not 0<=color<=3.3))
        if not eligible:continue
        predicted={b:scientific_predict(b,f) for b in BRANCHES}
        for branch,q in predicted.items():
            z=q[['tic_id','sector','correction','gaia_log_anchor','ref_count','pred_knn']].copy();z['branch']=branch;factors.append(z)
        products=pd.read_parquet(VARIABLE/'products.parquet');products=products[products.tic_id.eq(tic)]
        bins={};limits={}
        for p in products.itertuples():
            assert sha(p.path)==p.sha256
            with fits.open(p.path,memmap=False) as h:
                d=h[1].data;t=np.asarray(d['TIME'],float);sap=np.asarray(d['SAP_FLUX'],float);pdc=np.asarray(d['PDCSAP_FLUX'],float)
                initial=(d['QUALITY']==0)&np.isfinite(t)&np.isfinite(sap)&(sap>0)
                adjusted=(sap-(1-float(h[1].header['CROWDSAP']))*np.median(sap[initial]))/float(h[1].header['FLFRCSAP'])
                common=initial&np.isfinite(pdc)&(pdc>0)&np.isfinite(adjusted)&(adjusted>0)
                data=dict(time=t[common],sap=sap[common],adjusted_sap=adjusted[common],pdcsap=pdc[common])
                for branch in BRANCHES:
                    fac=float(predicted[branch].set_index('sector').loc[p.sector,'correction']);data[branch+'_hgb']=data[branch]/fac
                original_factor=float(pp.set_index('sector').loc[p.sector,'correction']);data['original_full_hgb']=data['pdcsap']/original_factor
                frame=pd.DataFrame(data);frame['bin']=np.floor(frame.time/.25).astype(int)
                g=frame.groupby('bin');b=g[list(data)].median();b['cadences']=g.size();b=b[b.cadences>=3].reset_index(drop=True)
                limits[p.sector]=(float(t[common].min()),float(t[common].max()));bins[p.sector]=b.copy()
                b['tic_id']=tic;b['sector']=p.sector;daily.append(b)
                attr.append(dict(tic_id=int(tic),sector=int(p.sector),total=len(d),sap_valid=int(initial.sum()),common=int(common.sum()),binned=int(b.cadences.sum()),qualified=False,tars_unknown=True))
        methods=[c for c in next(iter(bins.values())).columns if c not in ['time','cadences']]
        seq=sorted(bins)
        for sa,sb in zip(seq[:-1],seq[1:]):
            if sb!=sa+1:continue
            gap=limits[sb][0]-limits[sa][1]
            if not 0<gap<=3:continue
            a=bins[sa].copy();b=bins[sb].copy()
            for c in methods:a[c]=np.log(a[c]);b[c]=np.log(b[c])
            for window in [3.,5.,7.]:
                raw=jump(a,b,limits[sa][1],limits[sb][0],window,'pdcsap')
                if raw is None:continue
                if not pseudogaps(a,*limits[sa],gap,window) or not pseudogaps(b,*limits[sb],gap,window):continue
                for c in methods:
                    value=jump(a,b,limits[sa][1],limits[sb][0],window,c)
                    edges.append(dict(tic_id=int(tic),sector_left=sa,sector_right=sb,window_days=window,method=c,jump_log_pp=value,qualified=False,tars_unknown=True))
        print('Science',tic,'sectors',len(bins),flush=True)
    pd.DataFrame(support).to_csv(ROOT/'science_support.csv',index=False)
    pd.concat(factors).to_parquet(ROOT/'science_factors.parquet',index=False)
    b=pd.concat(daily,ignore_index=True);b.to_parquet(ROOT/'science_binned_curves.parquet',index=False)
    pd.DataFrame(attr).to_csv(ROOT/'science_attrition.csv',index=False)
    e=pd.DataFrame(edges);e.to_csv(ROOT/'science_boundaries.csv',index=False)
    s=e.groupby(['tic_id','window_days','method']).agg(boundaries=('jump_log_pp','count'),median_abs_jump_log_pp=('jump_log_pp',lambda x:float(np.median(abs(x))))).reset_index()
    s.to_csv(ROOT/'science_boundary_summary.csv',index=False)
    ranges=[]
    for tic,z in b.groupby('tic_id'):
        for method in methods:
            levels=z.groupby('sector')[method].median();normalized=100*(levels/levels.median()-1)
            ranges.append(dict(tic_id=int(tic),method=method,sectors=len(levels),sector_p90_minus_p10_pp=float(np.percentile(normalized,90)-np.percentile(normalized,10))))
    pd.DataFrame(ranges).to_csv(ROOT/'science_sector_ranges.csv',index=False)
    print(s[s.window_days.eq(5)].to_string(index=False),flush=True)

if __name__=='__main__':globals()[sys.argv[1]]()
