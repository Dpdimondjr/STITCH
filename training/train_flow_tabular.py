"""
Flow v2c: alpha=0 pure NLL, tabular+knn_100 context only (no spatial KNN features).
Avoids the NaN-imputation problem with sparse spatial features.
Context: BASE_COLS + THERMAL + PDC + SEC_AGG + MISC + knn_100 scalar = ~33D
"""
import sys, time, warnings; warnings.filterwarnings("ignore")
sys.stdout.reconfigure(line_buffering=True)
import numpy as np, pandas as pd
import torch, zuko
from torch.utils.data import DataLoader, TensorDataset
from scipy.spatial import cKDTree

_out_pt = next((s.split("=")[1] for s in sys.argv[1:] if s.startswith("--out=")), "stitch_nsf_v2c.pt")
SEED=20260915; MIN_SECTORS=5; MAX_TMAG=13; REF_CLIP=(0.5,1.5); KNN_K=100
KEYS=['sector','cam','ccd']
TRANSFORMS=8; HIDDEN=[256,256]; BINS=8; LR=1e-4; BATCH=4096; EPOCHS=200; PATIENCE=20
t0=time.time()
def log(m): print(m, flush=True)
torch.manual_seed(SEED); np.random.seed(SEED)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
log(f"Device: {device}  out={_out_pt}")

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
log(f"  {len(df):,} obs, {df.tic_id.nunique():,} TICs  T={time.time()-t0:.0f}s")

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

log("KNN-100..."); knn_tr=knn_median(ref_full,train_all); knn_vl=knn_median(ref_full,val_r); knn_te=knn_median(ref_full,test_r)
log(f"  Done  T={time.time()-t0:.0f}s")

BASE=['col','row','delta_sub_col','delta_sub_row','sector','cam','ccd',
      'tmag','crowdsap','cdpp1_0','pdcvar','jitter_rms','pdc_noi','pr_wght2','gaiarp']
THERMAL=['focal_plane_temp','focal_plane_temp_std','temp_x_col','temp_x_row','temp_x_crowdsap']
PDC_F=['pdc_corp','pdc_totp','pdctotp_x_crowdsap']
SEC_AGG=['sector_mean_loo','sector_std_loo','sector_frac_extreme']
MISC=['teff','bkg_rms','scatter_flag_frac','n_sectors_obs']
TAB_COLS=[c for c in BASE+THERMAL+PDC_F+SEC_AGG+MISC if c in train_all.columns]

def make_ctx(tab, knn_arr):
    X = np.hstack([tab[TAB_COLS].to_numpy(dtype=float), knn_arr.reshape(-1,1)])
    med = np.nanmedian(X, axis=0)
    X[np.isnan(X)] = np.take(med, np.where(np.isnan(X))[1])
    return X

X_tr_raw=make_ctx(train_all,knn_tr); X_vl_raw=make_ctx(val_r,knn_vl); X_te_raw=make_ctx(test_r,knn_te)
ctx_mean=X_tr_raw.mean(0).astype(np.float32); ctx_std=X_tr_raw.std(0).astype(np.float32)
ctx_std[ctx_std<1e-8]=1.0
X_tr=((X_tr_raw-ctx_mean)/ctx_std).astype(np.float32)
X_vl=((X_vl_raw-ctx_mean)/ctx_std).astype(np.float32)
X_te=((X_te_raw-ctx_mean)/ctx_std).astype(np.float32)
y_mean=float(train_all.flux_offset_loo.mean()); y_std=float(train_all.flux_offset_loo.std())
y_tr=((train_all.flux_offset_loo.values-y_mean)/y_std).astype(np.float32)
y_vl=((val_r.flux_offset_loo.values-y_mean)/y_std).astype(np.float32)
log(f"  Context: {X_tr.shape[1]}D  y_mean={y_mean:.6f}  y_std={y_std:.6f}")

flow=zuko.flows.NSF(features=1,context=X_tr.shape[1],transforms=TRANSFORMS,
                    hidden_features=HIDDEN,bins=BINS).to(device)
log(f"  Params: {sum(p.numel() for p in flow.parameters()):,}")
opt=torch.optim.Adam(flow.parameters(),lr=LR)
scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,patience=10,factor=0.5,min_lr=1e-6)

C_tr_t=torch.tensor(X_tr).to(device); y_tr_t=torch.tensor(y_tr).unsqueeze(-1).to(device)
C_vl_t=torch.tensor(X_vl).to(device); y_vl_t=torch.tensor(y_vl).unsqueeze(-1).to(device)
C_te_t=torch.tensor(X_te).to(device)
ds=TensorDataset(C_tr_t,y_tr_t); dl=DataLoader(ds,batch_size=BATCH,shuffle=True)

def flow_cv(part,C_t,n=200):
    flow.eval()
    with torch.no_grad():
        pred=flow(C_t).sample((n,)).squeeze(-1).mean(0).cpu().numpy()*y_std+y_mean
    rows=[]
    for tic,g in part.groupby('tic_id'):
        if g.sector.nunique()<MIN_SECTORS: continue
        c=g.sector_median.values/pred[g.index]; rows.append(c.std()/c.mean()*100)
    return float(np.median(rows)) if rows else float('nan')

log("\nTraining (pure NLL)...")
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
log(f"\n=== Results (best={best_epoch}, val_nll={best_nll:.4f}) ===")
cv_vl=flow_cv(val_r,C_vl_t); cv_te=flow_cv(test_r,C_te_t)
log(f"  val_CV={cv_vl:.4f}%  test_CV={cv_te:.4f}%")
log(f"  Production flow: 1.0494%  |  HGB: 0.8856%")

torch.save({'flow_config':{'features':1,'context':X_tr.shape[1],'transforms':TRANSFORMS,
    'hidden_features':HIDDEN,'bins':BINS},'model_state':best_state,
    'ctx_mean':ctx_mean,'ctx_std':ctx_std,'y_mean':y_mean,'y_std':y_std,
    'tab_cols':TAB_COLS,'sec_stats':sec_stats,
    'ref_spatial':ref_full[['tic_id','sector','cam','ccd','col','row','flux_offset_loo']].reset_index(drop=True)},
    _out_pt)
log(f"Saved → {_out_pt}  T={time.time()-t0:.0f}s")
