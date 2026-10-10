"""Cross-fitted hierarchical spatial-surface features over median-KNN residuals."""
from pathlib import Path
import json, time, sys
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split, KFold

sys.path.insert(0,str(Path(__file__).parent))
from residual_multiscale_experiment import spatial_features, scatter_table, tune_alpha

SEED=20260916; KEYS=['sector','cam','ccd']

def design(x,y):
    return np.column_stack([np.ones(len(x)),x,y,x*x,x*y,y*y,
                            x*x*x,x*x*y,x*y*y,y*y*y])

def fit_surfaces(reference,query):
    """Robust degree-3 prediction and geometry from an explicit reference bank."""
    q=query.reset_index(drop=True); out=np.full((len(q),7),np.nan,np.float32)
    groups=dict(tuple(reference.groupby(KEYS,sort=False)))
    for key,qg in q.groupby(KEYS,sort=False):
        r=groups.get(key)
        if r is None or len(r)<30: continue
        cx=float(r.col.mean()); cy=float(r.row.mean())
        sx=max(float(r.col.std()),1.); sy=max(float(r.row.std()),1.)
        x=(r.col.to_numpy()-cx)/sx; y=(r.row.to_numpy()-cy)/sy
        z=np.log(r.flux_offset_loo.to_numpy())
        X=design(x,y); keep=np.ones(len(r),bool)
        for _ in range(3):
            coef=np.linalg.lstsq(X[keep],z[keep],rcond=None)[0]
            residual=z-X@coef
            med=np.median(residual[keep]); mad=1.4826*np.median(np.abs(residual[keep]-med))
            if not np.isfinite(mad) or mad<1e-6: break
            keep=np.abs(residual-med)<3*mad
            if keep.sum()<20: keep[:]=True; break
        coef=np.linalg.lstsq(X[keep],z[keep],rcond=None)[0]
        residual=z-X@coef; scale=1.4826*np.median(np.abs(residual[keep]-np.median(residual[keep])))
        xx=(qg.col.to_numpy()-cx)/sx; yy=(qg.row.to_numpy()-cy)/sy
        pred=design(xx,yy)@coef
        # Analytic derivatives in normalized CCD coordinates.
        gx=coef[1]+2*coef[3]*xx+coef[4]*yy+3*coef[6]*xx**2+2*coef[7]*xx*yy+coef[8]*yy**2
        gy=coef[2]+coef[4]*xx+2*coef[5]*yy+coef[7]*xx**2+2*coef[8]*xx*yy+3*coef[9]*yy**2
        edge=np.minimum.reduce([qg.col.to_numpy(),qg.row.to_numpy(),
                                2200-qg.col.to_numpy(),2200-qg.row.to_numpy()])
        vals=np.column_stack([np.exp(pred),gx,gy,np.hypot(gx,gy),
                              np.full(len(qg),scale),np.hypot(xx,yy),edge])
        out[qg.index]=vals
    return pd.DataFrame(out,columns=['surface_pred','surface_grad_col','surface_grad_row',
        'surface_grad_mag','surface_resid_scale','surface_radius','edge_distance'])

def split(df):
    stars=df.groupby('tic_id').cam.agg(lambda x:x.mode().iloc[0]).rename('cam').reset_index()
    tr,tmp=train_test_split(stars,test_size=.2,stratify=stars.cam,random_state=42)
    va,te=train_test_split(tmp,test_size=.5,stratify=tmp.cam,random_state=42)
    return [set(x.tic_id) for x in [tr,va,te]]

def main():
    t0=time.time(); out=Path('eval/spatial_surface_results'); out.mkdir(exist_ok=True)
    d=pd.read_parquet('training_data_v3.parquet')
    d=d.dropna(subset=['col','row','flux_offset_loo'])
    d=d[d.flux_offset_loo.between(.85,1.15,inclusive='neither') & d.n_sectors_total.ge(4)&d.tmag.le(13)].reset_index(drop=True)
    train_ids,val_ids,test_ids=split(d); masks=[d.tic_id.isin(x) for x in [train_ids,val_ids,test_ids]]
    frames=[d[m].reset_index(drop=True) for m in masks]; train,val,test=frames
    print('KNN features',flush=True)
    knn=[spatial_features(train,x) for x in frames]
    print('Cross-fitted training surfaces',flush=True)
    surface_train=pd.DataFrame(np.nan,index=train.index,columns=['surface_pred','surface_grad_col','surface_grad_row','surface_grad_mag','surface_resid_scale','surface_radius','edge_distance'])
    star_ids=np.array(sorted(train.tic_id.unique())); kf=KFold(5,shuffle=True,random_state=SEED)
    for fit_idx,hold_idx in kf.split(star_ids):
        ref=train[train.tic_id.isin(set(star_ids[fit_idx]))]
        mask=train.tic_id.isin(set(star_ids[hold_idx])); surface_train.loc[mask]=fit_surfaces(ref,train[mask]).to_numpy()
    surfaces=[surface_train,fit_surfaces(train,val),fit_surfaces(train,test)]
    base=['col','row','delta_sub_col','delta_sub_row','sector','cam','ccd','tmag','crowdsap',
          'cdpp1_0','pdcvar','jitter_rms','pdc_noi','pr_wght2','gaiarp','focal_plane_temp']
    common=base+['knn20_median','knn20_mad','knn20_radius','reference_count']
    predictions={}; validation={}
    for label,extra in [('residual_k20',[]),('residual_surface',list(surfaces[0].columns))]:
        cols=common+extra
        X=[]
        for frame,k,s in zip(frames,knn,surfaces): X.append(pd.concat([frame[base],k,s],axis=1)[cols])
        y=train.flux_offset_loo.to_numpy()-knn[0].knn20_median.to_numpy(); good=np.isfinite(y)
        model=HistGradientBoostingRegressor(loss='absolute_error',learning_rate=.06,max_iter=250,
            max_leaf_nodes=31,min_samples_leaf=40,l2_regularization=.1,random_state=SEED,early_stopping=False)
        print('fit',label,len(cols),flush=True); model.fit(X[0].loc[good],y[good])
        pv=knn[1].knn20_median.to_numpy()+model.predict(X[1]); pt=knn[2].knn20_median.to_numpy()+model.predict(X[2])
        alpha,score=tune_alpha(val,pv); validation[label]={'alpha':alpha,'scatter':score}; predictions[label]=1+alpha*(pt-1)
    per=scatter_table(test,{'raw':np.ones(len(test)),'median_knn':knn[2].knn20_median.to_numpy(),**predictions})
    summary={c:{'scatter_pct':float(per[c].median()*100),'reduction_pct':float((1-per[c].median()/per.raw.median())*100),'worse_pct':float((per[c]>per.raw).mean()*100)} for c in per}
    payload={'validation':validation,'test':summary,'elapsed_seconds':time.time()-t0,'policy':'training-only references; 5-fold cross-fit training surfaces'}
    (out/'results.json').write_text(json.dumps(payload,indent=2)); per.to_csv(out/'per_star.csv'); print(json.dumps(payload,indent=2))

if __name__=='__main__': main()
