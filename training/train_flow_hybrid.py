"""
Flow v2d: HGB for point prediction + NSF for residual distribution.
HGB captures the mean (achieves 0.89% CV); flow learns p(log_resid | features).
Point estimate at inference = exp(hgb_logpred + flow_mean_resid) ≈ hgb_pred.
Uncertainty from flow std of residual.
"""
import sys, time, warnings; warnings.filterwarnings("ignore")
sys.stdout.reconfigure(line_buffering=True)
import numpy as np, pandas as pd
import torch, zuko
from torch.utils.data import DataLoader, TensorDataset
from scipy.spatial import cKDTree
from lightgbm import LGBMRegressor

SEED=20260915; MIN_SECTORS=5; MAX_TMAG=13; REF_CLIP=(0.5,1.5); KNN_K=100
TRANSFORMS=8; HIDDEN=[256,256]; BINS=8; LR=1e-4; BATCH=4096; EPOCHS=200; PATIENCE=20
t0=time.time()
def log(m): print(m, flush=True)
torch.manual_seed(SEED); np.random.seed(SEED)
device='cuda' if torch.cuda.is_available() else 'cpu'
log(f"Device: {device}")

df=pd.read_parquet('/Users/daviddimond/Documents/STITCH/training_data_v3.parquet',engine='pyarrow')
df['tic_id']=df['tic_id'].astype('int64'); df['sector']=df['sector'].astype('int64')
valid=(df.sector.between(1,100))&(df.cam.between(1,4))&(df.ccd.between(1,4))&\
      (df.tmag<=MAX_TMAG)&(df.gaiarp.notna())&(df.sector_median>0)&\
      (df.col.between(0,2200))&(df.row.between(0,2200))
df=df[valid]
sc=df.groupby('tic_id')['sector'].nunique(); df=df[df.tic_id.isin(sc[sc>=MIN_SECTORS].index)]
df['focal_plane_temp']=df['focal_plane_temp'].astype('float64')
df['focal_plane_temp_std']=df['focal_plane_temp_std'].astype('float64')
df['temp_x_col']=df['focal_plane_temp']*(df['col']/2200.0)
df['temp_x_row']=df['focal_plane_temp']*(df['row']/2200.0)
df['temp_x_crowdsap']=df['focal_plane_temp']*df['crowdsap']
for c in ['pdc_corp','pdc_totp','teff','bkg_rms','scatter_flag_frac']:
    df[c]=df[c].astype('float64') if c in df.columns else float('nan')
df['pdctotp_x_crowdsap']=df['pdc_totp']*df['crowdsap']
df['n_sectors_obs']=df.groupby('tic_id')['sector'].transform('nunique').astype(float)
df=df.reset_index(drop=True)
log(f"  {len(df):,} obs  T={time.time()-t0:.0f}s")

split=pd.read_csv('/Users/daviddimond/Documents/STITCH/eval/value_benchmark_results/split.csv')
def get_p(n): return set(split[split.partition==n].tic_id.astype('int64').values)
test_tics=get_p('test'); val_tics=get_p('validation')
ref_full=df[~df.tic_id.isin(test_tics)&df.flux_offset_loo.between(*REF_CLIP)].copy()
train_all=df[~df.tic_id.isin(test_tics)&~df.tic_id.isin(val_tics)].copy()
train_all=train_all[train_all.flux_offset_loo.between(*REF_CLIP)].reset_index(drop=True)
val_r=df[df.tic_id.isin(val_tics)].reset_index(drop=True)
test_r=df[df.tic_id.isin(test_tics)].reset_index(drop=True)

sec_stats=ref_full.groupby(['sector','cam','ccd'])['flux_offset_loo'].agg(
    sector_mean_loo='mean',sector_std_loo='std',
    sector_frac_extreme=lambda x:(np.abs(x-1)>0.02).mean()).reset_index()
for df_ in [train_all,val_r,test_r]:
    df_.drop(columns=['sector_mean_loo','sector_std_loo','sector_frac_extreme'],errors='ignore',inplace=True)
    tmp=df_.merge(sec_stats,on=['sector','cam','ccd'],how='left')
    df_['sector_mean_loo']=tmp['sector_mean_loo'].values
    df_['sector_std_loo']=tmp['sector_std_loo'].values
    df_['sector_frac_extreme']=tmp['sector_frac_extreme'].values

def knn_median(ref, query, k=KNN_K):
    result=np.ones(len(query)); query=query.reset_index(drop=True)
    for key,qg in query.groupby(['sector','cam','ccd']):
        r=ref[(ref.sector==key[0])&(ref.cam==key[1])&(ref.ccd==key[2])]
        if len(r)<2: continue
        coords=r[['col','row']].to_numpy(float); qcoords=qg[['col','row']].to_numpy(float)
        qids=qg.tic_id.to_numpy(); rids=r.tic_id.to_numpy()
        take=min(k+int(r.tic_id.value_counts().max()),len(r))
        dists,idxs=cKDTree(coords).query(qcoords,k=take)
        if take==1: dists,idxs=dists[:,None],idxs[:,None]
        loo=r.flux_offset_loo.to_numpy()
        for i,(q_id,nn) in enumerate(zip(qids,idxs)):
            keep=rids[nn]!=q_id; vals=loo[nn[keep][:k]]
            if len(vals)>=1: result[qg.index[i]]=np.median(vals)
    return result

def spatial_features(ref, query, ks=[10,50,100]):
    cols=[]; query=query.reset_index(drop=True)
    names=[f'knn{k}_mean' for k in ks]+[f'knn{k}_std' for k in ks]+[f'knn{k}_frac' for k in ks]
    out=np.full((len(query),len(names)),np.nan)
    for key,qg in query.groupby(['sector','cam','ccd']):
        r=ref[(ref.sector==key[0])&(ref.cam==key[1])&(ref.ccd==key[2])]
        if len(r)<2: continue
        coords=r[['col','row']].to_numpy(float); qcoords=qg[['col','row']].to_numpy(float)
        qids=qg.tic_id.to_numpy(); rids=r.tic_id.to_numpy()
        maxk=max(ks)+1; take=min(maxk+int(r.tic_id.value_counts().max()),len(r))
        dists,idxs=cKDTree(coords).query(qcoords,k=take)
        if take==1: dists,idxs=dists[:,None],idxs[:,None]
        loo=r.flux_offset_loo.to_numpy()
        for i,(q_id,nn) in enumerate(zip(qids,idxs)):
            keep=rids[nn]!=q_id; kept_loo=loo[nn[keep]]
            row_vals=[]
            for ki,k in enumerate(ks):
                vals=kept_loo[:k]
                if len(vals)<1: row_vals+=[np.nan,np.nan,np.nan]; continue
                row_vals+=[np.mean(vals),np.std(vals) if len(vals)>1 else 0.0,
                           np.mean(np.abs(vals-1)>0.02)]
            out[qg.index[i]]=row_vals
    return out, names

log("KNN-100..."); knn_tr=knn_median(ref_full,train_all); knn_vl=knn_median(ref_full,val_r); knn_te=knn_median(ref_full,test_r)
log("Spatial..."); sp_tr,sp_names=spatial_features(ref_full,train_all); sp_vl,_=spatial_features(ref_full,val_r); sp_te,_=spatial_features(ref_full,test_r)
log(f"  Done  T={time.time()-t0:.0f}s")

BASE=['col','row','delta_sub_col','delta_sub_row','sector','cam','ccd',
      'tmag','crowdsap','cdpp1_0','pdcvar','jitter_rms','pdc_noi','pr_wght2','gaiarp']
THERMAL=['focal_plane_temp','focal_plane_temp_std','temp_x_col','temp_x_row','temp_x_crowdsap']
PDC_F=['pdc_corp','pdc_totp','pdctotp_x_crowdsap']
SEC_AGG=['sector_mean_loo','sector_std_loo','sector_frac_extreme']
MISC=['teff','bkg_rms','scatter_flag_frac','n_sectors_obs']
TAB_COLS=[c for c in BASE+THERMAL+PDC_F+SEC_AGG+MISC if c in train_all.columns]

def make_X_hgb(tab, knn_arr, sp_arr):
    tab_arr=tab[TAB_COLS].to_numpy(dtype=float)
    return np.hstack([tab_arr, knn_arr.reshape(-1,1), sp_arr])

X_hgb_tr=make_X_hgb(train_all,knn_tr,sp_tr)
X_hgb_vl=make_X_hgb(val_r,knn_vl,sp_vl)
X_hgb_te=make_X_hgb(test_r,knn_te,sp_te)
y_hgb_tr=np.log(train_all.flux_offset_loo.values)

# Train HGB in 5 rounds with LOO-style progressive beta scaling
log("Training HGB (5 rounds)...")
HGB_PARAMS=dict(n_estimators=500,learning_rate=0.05,max_depth=6,num_leaves=63,
                min_child_samples=20,subsample=0.8,colsample_bytree=0.8,random_state=SEED,
                n_jobs=-1,verbose=-1)

all_preds_vl=[]; all_preds_te=[]
hgb_pred_vl=np.ones(len(val_r)); hgb_pred_te=np.ones(len(test_r))
hgb_pred_tr=np.ones(len(train_all))

def cv_from_pred(part, pred, label=''):
    rows=[]
    for tic,g in part.groupby('tic_id'):
        if g.sector.nunique()<MIN_SECTORS: continue
        c=g.sector_median.values/pred[g.index]; rows.append(c.std()/c.mean()*100)
    v=float(np.median(rows)) if rows else float('nan')
    if label: log(f"  {label}  {v:.4f}%")
    return v

best_cv_vl=1e9; best_beta=1.0; best_hgb=None
for rnd in range(5):
    y_target=np.log(train_all.flux_offset_loo.values)-rnd*0.1*np.log(np.maximum(hgb_pred_tr,0.5))
    m=LGBMRegressor(**HGB_PARAMS); m.fit(X_hgb_tr,y_target,eval_set=[(X_hgb_vl,np.log(val_r.flux_offset_loo.values))],callbacks=[])
    raw_vl=np.exp(m.predict(X_hgb_vl)); raw_te=np.exp(m.predict(X_hgb_te)); raw_tr=np.exp(m.predict(X_hgb_tr))
    betas=np.linspace(0.5,2.0,16)
    cv_vals=[cv_from_pred(val_r, raw_vl**b * knn_vl**(1-b)) for b in betas]
    b_best=float(betas[np.argmin(cv_vals)])
    pred_vl_b=raw_vl**b_best*knn_vl**(1-b_best); pred_te_b=raw_te**b_best*knn_te**(1-b_best)
    pred_tr_b=raw_tr**b_best*knn_tr**(1-b_best)
    cv_v=cv_from_pred(val_r,pred_vl_b)
    log(f"  Round {rnd+1}/5  beta={b_best:.2f}  val_CV={cv_v:.4f}%  T={time.time()-t0:.0f}s")
    if cv_v<best_cv_vl: best_cv_vl=cv_v; best_beta=b_best; best_hgb=m
    hgb_pred_tr=pred_tr_b; hgb_pred_vl=pred_vl_b; hgb_pred_te=pred_te_b

raw_vl_f=np.exp(best_hgb.predict(X_hgb_vl)); raw_te_f=np.exp(best_hgb.predict(X_hgb_te)); raw_tr_f=np.exp(best_hgb.predict(X_hgb_tr))
hgb_pred_vl=raw_vl_f**best_beta*knn_vl**(1-best_beta)
hgb_pred_te=raw_te_f**best_beta*knn_te**(1-best_beta)
hgb_pred_tr=raw_tr_f**best_beta*knn_tr**(1-best_beta)

log(f"\nHGB best  val_CV={best_cv_vl:.4f}%  beta={best_beta:.2f}")
cv_from_pred(test_r, hgb_pred_te, 'HGB test_CV')

# Build flow context: tabular + knn_100 + residual signal
def make_ctx_flow(tab, knn_arr, sp_arr):
    X=make_X_hgb(tab, knn_arr, sp_arr)
    med=np.nanmedian(X,axis=0); X[np.isnan(X)]=np.take(med,np.where(np.isnan(X))[1])
    return X

X_flow_tr=make_ctx_flow(train_all,knn_tr,sp_tr)
X_flow_vl=make_ctx_flow(val_r,knn_vl,sp_vl)
X_flow_te=make_ctx_flow(test_r,knn_te,sp_te)

ctx_mean=X_flow_tr.mean(0).astype(np.float32); ctx_std=X_flow_tr.std(0).astype(np.float32)
ctx_std[ctx_std<1e-8]=1.0
X_tr_n=((X_flow_tr-ctx_mean)/ctx_std).astype(np.float32)
X_vl_n=((X_flow_vl-ctx_mean)/ctx_std).astype(np.float32)
X_te_n=((X_flow_te-ctx_mean)/ctx_std).astype(np.float32)

# Flow target: log residual from HGB
log_resid_tr=(np.log(train_all.flux_offset_loo.values)-np.log(np.maximum(hgb_pred_tr,0.5))).astype(np.float32)
log_resid_vl=(np.log(val_r.flux_offset_loo.values)-np.log(np.maximum(hgb_pred_vl,0.5))).astype(np.float32)
resid_mean=float(log_resid_tr.mean()); resid_std=float(log_resid_tr.std())
y_tr=((log_resid_tr-resid_mean)/resid_std).astype(np.float32)
y_vl=((log_resid_vl-resid_mean)/resid_std).astype(np.float32)
log(f"  Residual target: mean={resid_mean:.5f}  std={resid_std:.5f}  (should be ~0, small)")
log(f"  Flow context: {X_tr_n.shape[1]}D")

flow=zuko.flows.NSF(features=1,context=X_tr_n.shape[1],transforms=TRANSFORMS,
                    hidden_features=HIDDEN,bins=BINS).to(device)
log(f"  Flow params: {sum(p.numel() for p in flow.parameters()):,}")
opt=torch.optim.Adam(flow.parameters(),lr=LR)
scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,patience=10,factor=0.5,min_lr=1e-6)

C_tr_t=torch.tensor(X_tr_n).to(device); y_tr_t=torch.tensor(y_tr).unsqueeze(-1).to(device)
C_vl_t=torch.tensor(X_vl_n).to(device); y_vl_t=torch.tensor(y_vl).unsqueeze(-1).to(device)
C_te_t=torch.tensor(X_te_n).to(device)
ds=TensorDataset(C_tr_t,y_tr_t); dl=DataLoader(ds,batch_size=BATCH,shuffle=True)

def hybrid_cv(part, hgb_arr, C_t, n=200):
    flow.eval()
    with torch.no_grad():
        samps=flow(C_t).sample((n,)).squeeze(-1).cpu().numpy()
    mean_resid=samps.mean(0)*resid_std+resid_mean
    pred=hgb_arr*np.exp(mean_resid)
    rows=[]
    for tic,g in part.groupby('tic_id'):
        if g.sector.nunique()<MIN_SECTORS: continue
        c=g.sector_median.values/pred[g.index]; rows.append(c.std()/c.mean()*100)
    return float(np.median(rows)) if rows else float('nan')

log("\nTraining flow on HGB residuals (pure NLL)...")
best_nll=float('inf'); best_epoch=0; patience_cnt=0; best_state=None
for epoch in range(1,EPOCHS+1):
    flow.train(); nll_sum=0; nb=0
    for C_b,y_b in dl:
        opt.zero_grad(); loss=-flow(C_b).log_prob(y_b).mean(); loss.backward(); opt.step()
        nll_sum+=loss.item(); nb+=1
    flow.eval()
    with torch.no_grad(): val_nll=-flow(C_vl_t).log_prob(y_vl_t).mean().item()
    scheduler.step(val_nll)
    if val_nll<best_nll:
        best_nll=val_nll; best_epoch=epoch; patience_cnt=0
        best_state={k:v.cpu().clone() for k,v in flow.state_dict().items()}
    else: patience_cnt+=1
    if epoch%10==0 or epoch<=5:
        log(f"  Epoch {epoch:3d}  train={nll_sum/nb:.4f}  val={val_nll:.4f}  best={best_nll:.4f}  T={time.time()-t0:.0f}s")
    if patience_cnt>=PATIENCE:
        log(f"  Early stop epoch {epoch} (best={best_epoch})"); break

flow.load_state_dict(best_state); flow.eval()
log(f"\n=== Results ===")
cv_vl=hybrid_cv(val_r, hgb_pred_vl, C_vl_t)
cv_te=hybrid_cv(test_r, hgb_pred_te, C_te_t)
log(f"  val_CV={cv_vl:.4f}%  test_CV={cv_te:.4f}%")
log(f"  HGB alone:      val_CV={best_cv_vl:.4f}%  (flow mean_resid≈0 → should match)")
log(f"  Production flow: 1.0494%  |  HGB ceiling: 0.8856%")

torch.save({'flow_config':{'features':1,'context':X_tr_n.shape[1],'transforms':TRANSFORMS,
    'hidden_features':HIDDEN,'bins':BINS},'model_state':best_state,
    'ctx_mean':ctx_mean,'ctx_std':ctx_std,'resid_mean':resid_mean,'resid_std':resid_std,
    'tab_cols':TAB_COLS,'sec_stats':sec_stats,'best_beta':best_beta,
    'ref_spatial':ref_full[['tic_id','sector','cam','ccd','col','row','flux_offset_loo']].reset_index(drop=True)},
    'stitch_nsf_v2d.pt')
log(f"Saved → stitch_nsf_v2d.pt  T={time.time()-t0:.0f}s")
