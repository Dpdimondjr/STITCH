"""
Flow v3b: Production features (18D) + per-star CV consistency loss.
Batches groups entire stars (all sectors) so loss can penalize
sector-to-sector inconsistency within each star.
Loss = NLL + lambda_cv * mean_star(std(m/pred) / mean(m/pred))
"""
import sys, time, warnings; warnings.filterwarnings("ignore")
sys.stdout.reconfigure(line_buffering=True)
import numpy as np, pandas as pd
import torch, zuko
from torch.utils.data import DataLoader, Sampler

SEED=20260915; MIN_SECTORS=5; MAX_TMAG=13
TRANSFORMS=8; HIDDEN=[128,128]; BINS=8; LR=1e-4
STARS_PER_BATCH=256; EPOCHS=300; PATIENCE=25; LAMBDA_CV=1.0
t0=time.time()
def log(m): print(m, flush=True)
torch.manual_seed(SEED); np.random.seed(SEED)
device='cuda' if torch.cuda.is_available() else 'cpu'
log(f"Device: {device}  lambda_cv={LAMBDA_CV}")

df=pd.read_parquet('/Users/daviddimond/Documents/STITCH/training_data_v3.parquet',engine='pyarrow')
df['tic_id']=df['tic_id'].astype('int64'); df['sector']=df['sector'].astype('int64')
valid=(df.sector.between(1,100))&(df.cam.between(1,4))&(df.ccd.between(1,4))&\
      (df.tmag<=MAX_TMAG)&(df.gaiarp.notna())&(df.sector_median>0)&\
      (df.col.between(0,2200))&(df.row.between(0,2200))
df=df[valid]
sc=df.groupby('tic_id')['sector'].nunique(); df=df[df.tic_id.isin(sc[sc>=MIN_SECTORS].index)]
df=df.reset_index(drop=True)
tars=pd.read_csv('/Users/daviddimond/Documents/STITCH/tars_quiet_tics_v2.csv')
tars_tics=set(tars.iloc[:,0].astype('int64').values)
split=pd.read_csv('/Users/daviddimond/Documents/STITCH/eval/value_benchmark_results/split.csv')
def get_p(n): return set(split[split.partition==n].tic_id.astype('int64').values)
test_tics=get_p('test'); val_tics=get_p('validation')
train_all=df[~df.tic_id.isin(test_tics)&~df.tic_id.isin(val_tics)&
              df.tic_id.isin(tars_tics)&df.flux_offset_loo.between(0.5,1.5)].reset_index(drop=True)
val_r=df[df.tic_id.isin(val_tics)].reset_index(drop=True)
test_r=df[df.tic_id.isin(test_tics)].reset_index(drop=True)
log(f"Train: {len(train_all):,} obs  {train_all.tic_id.nunique():,} TICs  y_std={train_all.flux_offset_loo.std():.5f}")

CONT=['col','row','delta_sub_col','delta_sub_row','sector','tmag','crowdsap',
      'cdpp1_0','pdcvar','jitter_rms']
for c in CONT:
    for d in [train_all,val_r,test_r]:
        if d[c].isna().any(): d[c]=d[c].fillna(float(train_all[c].median()))
means=train_all[CONT].mean().astype(np.float32); stds=train_all[CONT].std().replace(0,1).astype(np.float32)

def build_ctx(df_):
    cont=((df_[CONT]-means)/stds).values.astype(np.float32)
    cam_oh=pd.get_dummies(df_.cam.astype(int),prefix='cam').reindex(
        columns=['cam_1','cam_2','cam_3','cam_4'],fill_value=0).values.astype(np.float32)
    ccd_oh=pd.get_dummies(df_.ccd.astype(int),prefix='ccd').reindex(
        columns=['ccd_1','ccd_2','ccd_3','ccd_4'],fill_value=0).values.astype(np.float32)
    return np.hstack([cont,cam_oh,ccd_oh])

C_tr_np=build_ctx(train_all); C_vl_np=build_ctx(val_r); C_te_np=build_ctx(test_r)
y_mean=float(train_all.flux_offset_loo.mean()); y_std=float(train_all.flux_offset_loo.std())
y_tr_np=((train_all.flux_offset_loo.values-y_mean)/y_std).astype(np.float32)
y_vl_np=((val_r.flux_offset_loo.values-y_mean)/y_std).astype(np.float32)
sm_tr=train_all.sector_median.values.astype(np.float32)  # for CV loss
log(f"Context: {C_tr_np.shape[1]}D  y_mean={y_mean:.6f}  y_std={y_std:.6f}")

# ── Star-grouped sampler ───────────────────────────────────────────────────────
train_star_ids=train_all.tic_id.values
unique_stars=np.unique(train_star_ids)
star_to_idx={s:np.where(train_star_ids==s)[0] for s in unique_stars}

class StarGroupSampler(Sampler):
    def __init__(self, stars, star_to_idx, stars_per_batch):
        self.stars=stars; self.s2i=star_to_idx; self.spb=stars_per_batch
    def __iter__(self):
        perm=np.random.permutation(self.stars)
        for i in range(0,len(perm),self.spb):
            batch_stars=perm[i:i+self.spb]
            idxs=np.concatenate([self.s2i[s] for s in batch_stars])
            yield idxs.tolist()
    def __len__(self): return (len(self.stars)+self.spb-1)//self.spb

C_tr_t=torch.tensor(C_tr_np).to(device)
y_tr_t=torch.tensor(y_tr_np).to(device)
sm_tr_t=torch.tensor(sm_tr).to(device)
star_id_t=torch.tensor(train_star_ids).to(device)

C_vl_t=torch.tensor(C_vl_np).to(device)
y_vl_t=torch.tensor(y_vl_np).unsqueeze(-1).to(device)
C_te_t=torch.tensor(C_te_np).to(device)

sampler=StarGroupSampler(unique_stars, star_to_idx, STARS_PER_BATCH)

flow=zuko.flows.NSF(features=1,context=C_tr_np.shape[1],transforms=TRANSFORMS,
                    hidden_features=HIDDEN,bins=BINS).to(device)
log(f"Params: {sum(p.numel() for p in flow.parameters()):,}")
opt=torch.optim.Adam(flow.parameters(),lr=LR)
sched=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,patience=12,factor=0.5,min_lr=1e-6)

def flow_cv(part,C_t,n=300,K=5.0):
    flow.eval()
    with torch.no_grad():
        samps=flow(C_t).sample((n,)).squeeze(-1).cpu().numpy()
    pred_raw=samps.mean(0)*y_std+y_mean; pred_std_=samps.std(0)*y_std
    w=1/(1+K*pred_std_); pred=1.0*(1-w)+pred_raw*w
    rows=[]
    for tic,g in part.groupby('tic_id'):
        if g.sector.nunique()<MIN_SECTORS: continue
        c=g.sector_median.values/pred[g.index]; rows.append(c.std()/c.mean()*100)
    return float(np.median(rows)) if rows else float('nan')

log(f"\nTraining (NLL + per-star CV loss, lambda={LAMBDA_CV})...")
best_nll=float('inf'); best_epoch=0; pc=0; best_state=None
for epoch in range(1,EPOCHS+1):
    flow.train(); nll_sum=0; cv_sum=0; nb=0
    for idx_list in sampler:
        idx=torch.tensor(idx_list,device=device)
        C_b=C_tr_t[idx]; y_b=y_tr_t[idx].unsqueeze(-1)
        sm_b=sm_tr_t[idx]; sids=star_id_t[idx]
        opt.zero_grad()
        nll=-flow(C_b).log_prob(y_b).mean()
        # Per-star CV consistency term
        with torch.no_grad():
            pred_raw_b=flow(C_b).sample((16,)).squeeze(-1).mean(0)*y_std+y_mean
        cv_pen=torch.tensor(0.0,device=device)
        n_stars_with_2=0
        for s in sids.unique():
            mask=sids==s
            if mask.sum()<2: continue
            ratios=sm_b[mask]/pred_raw_b[mask]
            cv_pen=cv_pen+ratios.std()/ratios.mean().clamp(min=1e-6)
            n_stars_with_2+=1
        if n_stars_with_2>0: cv_pen=cv_pen/n_stars_with_2
        loss=nll+LAMBDA_CV*cv_pen
        loss.backward(); opt.step()
        nll_sum+=nll.item(); cv_sum+=cv_pen.item(); nb+=1
    flow.eval()
    with torch.no_grad(): vnll=-flow(C_vl_t).log_prob(y_vl_t).mean().item()
    sched.step(vnll)
    if vnll<best_nll: best_nll=vnll; best_epoch=epoch; pc=0; best_state={k:v.cpu().clone() for k,v in flow.state_dict().items()}
    else: pc+=1
    if epoch%10==0 or epoch<=5:
        log(f"  Epoch {epoch:3d}  nll={nll_sum/nb:.4f}  cv_pen={cv_sum/nb:.4f}  val={vnll:.4f}  best={best_nll:.4f}  T={time.time()-t0:.0f}s")
    if pc>=PATIENCE: log(f"  Early stop epoch {epoch} (best={best_epoch})"); break

flow.load_state_dict(best_state); flow.eval()
cv_vl=flow_cv(val_r,C_vl_t); cv_te=flow_cv(test_r,C_te_t)
log(f"\n=== v3b Results ===")
log(f"  val_CV={cv_vl:.4f}%  test_CV={cv_te:.4f}%")
log(f"  Production: 1.0494%  |  HGB: 0.8856%")

torch.save({'flow_config':{'features':1,'context':C_tr_np.shape[1],'transforms':TRANSFORMS,
    'hidden_features':HIDDEN,'bins':BINS},'model_state':best_state,
    'means':means,'stds':stds,'y_mean':y_mean,'y_std':y_std,
    'continuous_cols':CONT,'cam_cols':['cam_1','cam_2','cam_3','cam_4'],
    'ccd_cols':['ccd_1','ccd_2','ccd_3','ccd_4']},'stitch_nsf_v3b.pt')
log(f"Saved → stitch_nsf_v3b.pt  T={time.time()-t0:.0f}s")
