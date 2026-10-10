"""
STITCH improved normalizing flow.

Changes from production stitch_nsf.pt (18D context):
  - Context: 65D (BASE_COLS + THERMAL + PDC + SEC_AGG + MISC + knn_100 + spatial)
  - The knn_100 spatial median is included as a context feature (not subtracted from target)
  - Mixed objective: NLL + alpha * MAE(mean_pred, y)  (default alpha=0.3)
  - Larger hidden layers: [256, 256] instead of [128, 128]
  - Target unchanged: flux_offset_loo directly (y_mean≈1, y_std≈0.02)

Saved checkpoint is drop-in compatible with infer_star.py (same keys).
Spatial features (knn_100, knn_5/10/20/50) need the reference set at inference
time — saved as ref_spatial.parquet alongside the model.

Usage:
  python3 training/train_flow_improved.py
  python3 training/train_flow_improved.py --alpha=0.3 --out=stitch_nsf_v2.pt
"""
import sys, time, warnings; warnings.filterwarnings("ignore")
sys.stdout.reconfigure(line_buffering=True)

import numpy as np, pandas as pd
import torch, zuko
from torch.utils.data import DataLoader, TensorDataset
from scipy.spatial import cKDTree

# ── Args ─────────────────────────────────────────────────────────────────────
_alpha  = float(next((s.split("=")[1] for s in sys.argv[1:] if s.startswith("--alpha=")), "0.3"))
_out_pt = next((s.split("=")[1] for s in sys.argv[1:] if s.startswith("--out=")),
               "stitch_nsf_v2.pt")

SEED=20260915; MIN_SECTORS=5; MAX_TMAG=13; REF_CLIP=(0.5,1.5); KNN_K=100
KEYS=['sector','cam','ccd']; KS=[5,10,20,50]
TRANSFORMS=8; HIDDEN=[256,256]; BINS=8
LR=1e-4; BATCH=4096; EPOCHS=200; PATIENCE=20
t0=time.time()

def log(m): print(m, flush=True)
torch.manual_seed(SEED); np.random.seed(SEED)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
log(f"Device: {device}  alpha={_alpha}  out={_out_pt}")

# ── Load & filter ─────────────────────────────────────────────────────────────
log("Loading data...")
df=pd.read_parquet('/Users/daviddimond/Documents/STITCH/training_data_v3.parquet',engine='pyarrow')
df['tic_id']=df['tic_id'].astype('int64'); df['sector']=df['sector'].astype('int64')
valid=(df.sector.between(1,100))&(df.cam.between(1,4))&(df.ccd.between(1,4))&\
      (df.tmag<=MAX_TMAG)&(df.gaiarp.notna())&(df.sector_median>0)&\
      (df.col.between(0,2200))&(df.row.between(0,2200))
df=df[valid]
sc=df.groupby('tic_id')['sector'].nunique(); df=df[df.tic_id.isin(sc[sc>=MIN_SECTORS].index)]

# Engineer features
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

# ── Split ─────────────────────────────────────────────────────────────────────
split=pd.read_csv('/Users/daviddimond/Documents/STITCH/eval/value_benchmark_results/split.csv')
def get_p(n): return set(split[split.partition==n].tic_id.astype('int64').values)
test_tics=get_p('test'); val_tics=get_p('validation')
ref_full=df[~df.tic_id.isin(test_tics)&df.flux_offset_loo.between(*REF_CLIP)].copy()
train_all=df[~df.tic_id.isin(test_tics)&~df.tic_id.isin(val_tics)].copy()
train_all=train_all[train_all.flux_offset_loo.between(*REF_CLIP)].reset_index(drop=True)
val_r=df[df.tic_id.isin(val_tics)].reset_index(drop=True)
test_r=df[df.tic_id.isin(test_tics)].reset_index(drop=True)
log(f"  train={len(train_all)}, val={len(val_r)}, test={len(test_r)}")

# ── Sector aggregates ─────────────────────────────────────────────────────────
sec_stats=ref_full.groupby(['sector','cam','ccd'])['flux_offset_loo'].agg(
    sector_mean_loo='mean', sector_std_loo='std',
    sector_frac_extreme=lambda x:(np.abs(x-1)>0.02).mean()).reset_index()
for df_ in [train_all,val_r,test_r]:
    df_.drop(columns=['sector_mean_loo','sector_std_loo','sector_frac_extreme'],errors='ignore',inplace=True)
    tmp=df_.merge(sec_stats,on=['sector','cam','ccd'],how='left')
    df_['sector_mean_loo']=tmp['sector_mean_loo'].values
    df_['sector_std_loo']=tmp['sector_std_loo'].values
    df_['sector_frac_extreme']=tmp['sector_frac_extreme'].values
log(f"  Sector aggregates  T={time.time()-t0:.0f}s")

# ── KNN-100 + spatial features ────────────────────────────────────────────────
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

def spatial_features(ref, query):
    ref=ref.reset_index(drop=True); query=query.reset_index(drop=True)
    cols=[f'knn{k}_{s}' for k in KS for s in ['mean','median','mad','radius']]+['reference_count']
    out=np.full((len(query),len(cols)),np.nan); out[:,-1]=0
    groups={key:grp for key,grp in ref.groupby(KEYS,sort=False)}
    for key,qg in query.groupby(KEYS,sort=False):
        r=groups.get(key)
        if r is None or len(r)<2: continue
        coords=r[['col','row']].to_numpy(float); ids=r.tic_id.to_numpy(); vals=r.flux_offset_loo.to_numpy()
        qcoords=qg[['col','row']].to_numpy(float); qids=qg.tic_id.to_numpy()
        extra=int(r.tic_id.value_counts().max()); take=min(max(KS)+extra,len(r))
        dists,idxs=cKDTree(coords).query(qcoords,k=take)
        if take==1: dists,idxs=dists[:,None],idxs[:,None]
        outpos=qg.index.to_numpy()
        sel=np.full((len(qg),max(KS)),-1,dtype=np.int64); sel_d=np.full((len(qg),max(KS)),np.nan)
        for row,(q_id,nn,dd) in enumerate(zip(qids,idxs,dists)):
            keep=ids[nn]!=q_id; nn2,dd2=nn[keep][:max(KS)],dd[keep][:max(KS)]
            sel[row,:len(nn2)]=nn2; sel_d[row,:len(dd2)]=dd2
        count=(sel>=0).sum(axis=1); out[outpos,-1]=count
        for ki,k in enumerate(KS):
            good=count>=k
            if not good.any(): continue
            x=vals[sel[good,:k]]
            med=np.median(x,axis=1); base=ki*4
            out[outpos[good],base]=x.mean(axis=1); out[outpos[good],base+1]=med
            out[outpos[good],base+2]=np.median(np.abs(x-med[:,None]),axis=1)
            out[outpos[good],base+3]=sel_d[good,k-1]
    return pd.DataFrame(out,index=query.index,columns=cols)

log("KNN-100...")
knn_tr=knn_median(ref_full,train_all); knn_vl=knn_median(ref_full,val_r); knn_te=knn_median(ref_full,test_r)
log("Spatial features...")
sp_tr=spatial_features(ref_full,train_all); sp_vl=spatial_features(ref_full,val_r); sp_te=spatial_features(ref_full,test_r)
log(f"  Done  T={time.time()-t0:.0f}s")

# ── Build feature matrix ──────────────────────────────────────────────────────
BASE=['col','row','delta_sub_col','delta_sub_row','sector','cam','ccd',
      'tmag','crowdsap','cdpp1_0','pdcvar','jitter_rms','pdc_noi','pr_wght2','gaiarp']
THERMAL=['focal_plane_temp','focal_plane_temp_std','temp_x_col','temp_x_row','temp_x_crowdsap']
PDC_F=['pdc_corp','pdc_totp','pdctotp_x_crowdsap']
SEC_AGG=['sector_mean_loo','sector_std_loo','sector_frac_extreme']
MISC=['teff','bkg_rms','scatter_flag_frac','n_sectors_obs']
TAB_COLS=[c for c in BASE+THERMAL+PDC_F+SEC_AGG+MISC if c in train_all.columns]

def make_context(tab, knn_arr, sp_df):
    tab_vals = tab[TAB_COLS].to_numpy(dtype=float)
    knn_col  = knn_arr.reshape(-1, 1)
    sp_vals  = sp_df.to_numpy(dtype=float)
    return np.hstack([tab_vals, knn_col, sp_vals])

X_tr_raw = make_context(train_all, knn_tr, sp_tr)
X_vl_raw = make_context(val_r,     knn_vl, sp_vl)
X_te_raw = make_context(test_r,    knn_te, sp_te)
log(f"  Context shape: {X_tr_raw.shape[1]}D  T={time.time()-t0:.0f}s")

# Impute NaN with column medians from training
col_medians = np.nanmedian(X_tr_raw, axis=0)
for X in [X_tr_raw, X_vl_raw, X_te_raw]:
    nan_mask = np.isnan(X)
    X[nan_mask] = np.take(col_medians, np.where(nan_mask)[1])

# Standardise context with training stats
ctx_mean = X_tr_raw.mean(axis=0).astype(np.float32)
ctx_std  = X_tr_raw.std(axis=0).astype(np.float32)
ctx_std[ctx_std < 1e-8] = 1.0
X_tr = ((X_tr_raw - ctx_mean) / ctx_std).astype(np.float32)
X_vl = ((X_vl_raw - ctx_mean) / ctx_std).astype(np.float32)
X_te = ((X_te_raw - ctx_mean) / ctx_std).astype(np.float32)

# Target: flux_offset_loo (standardised)
y_mean = float(train_all.flux_offset_loo.mean())
y_std  = float(train_all.flux_offset_loo.std())
y_tr = ((train_all.flux_offset_loo.values - y_mean) / y_std).astype(np.float32)
y_vl = ((val_r.flux_offset_loo.values      - y_mean) / y_std).astype(np.float32)

log(f"  y_mean={y_mean:.6f}  y_std={y_std:.6f}")
log(f"  Context dim: {X_tr.shape[1]}")

# ── Build flow ────────────────────────────────────────────────────────────────
log(f"\nBuilding flow: transforms={TRANSFORMS} hidden={HIDDEN} bins={BINS}...")
flow = zuko.flows.NSF(
    features        = 1,
    context         = X_tr.shape[1],
    transforms      = TRANSFORMS,
    hidden_features = HIDDEN,
    bins            = BINS,
).to(device)
n_params = sum(p.numel() for p in flow.parameters())
log(f"  Parameters: {n_params:,}")

opt = torch.optim.Adam(flow.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=10, factor=0.5, min_lr=1e-6)

C_tr_t = torch.tensor(X_tr).to(device)
y_tr_t = torch.tensor(y_tr).unsqueeze(-1).to(device)
C_vl_t = torch.tensor(X_vl).to(device)
y_vl_t = torch.tensor(y_vl).unsqueeze(-1).to(device)
C_te_t = torch.tensor(X_te).to(device)

ds = TensorDataset(C_tr_t, y_tr_t)
dl = DataLoader(ds, batch_size=BATCH, shuffle=True)

# ── CV metric for flow ────────────────────────────────────────────────────────
def flow_cv(part, C_t, n_samples=200):
    flow.eval()
    with torch.no_grad():
        samples = flow(C_t).sample((n_samples,)).squeeze(-1)  # (n_samples, N)
        pred_z  = samples.mean(0).cpu().numpy()
    pred = pred_z * y_std + y_mean
    rows = []
    for tic, g in part.groupby('tic_id'):
        if g.sector.nunique() < MIN_SECTORS: continue
        c = g.sector_median.values / pred[g.index]
        rows.append(c.std() / c.mean() * 100)
    return float(np.median(rows)) if rows else float('nan')

# ── Training loop ─────────────────────────────────────────────────────────────
log(f"\nTraining (alpha={_alpha})...")
best_val_nll = float('inf'); best_epoch = 0; patience_count = 0
best_state = None

for epoch in range(1, EPOCHS + 1):
    flow.train()
    epoch_nll = 0.0; epoch_mae = 0.0; n_batches = 0
    for C_b, y_b in dl:
        opt.zero_grad()
        nll = -flow(C_b).log_prob(y_b).mean()
        if _alpha > 0:
            # MAE on mean prediction (encourages mode alignment with ground truth)
            with torch.no_grad():
                samps = flow(C_b).sample((32,)).squeeze(-1)
            mae = (samps.mean(0) - y_b.squeeze(-1)).abs().mean()
            loss = nll + _alpha * mae
        else:
            loss = nll; mae = torch.tensor(0.0)
        loss.backward(); opt.step()
        epoch_nll += nll.item(); epoch_mae += mae.item(); n_batches += 1

    flow.eval()
    with torch.no_grad():
        val_nll = -flow(C_vl_t).log_prob(y_vl_t).mean().item()
    scheduler.step(val_nll)

    if val_nll < best_val_nll:
        best_val_nll = val_nll; best_epoch = epoch
        best_state = {k: v.cpu().clone() for k, v in flow.state_dict().items()}
        patience_count = 0
    else:
        patience_count += 1

    if epoch % 10 == 0 or epoch <= 5:
        log(f"  Epoch {epoch:3d}  train_nll={epoch_nll/n_batches:.4f}  "
            f"val_nll={val_nll:.4f}  best={best_val_nll:.4f}  T={time.time()-t0:.0f}s")

    if patience_count >= PATIENCE:
        log(f"  Early stop at epoch {epoch} (best={best_epoch})")
        break

# ── Evaluate best model ───────────────────────────────────────────────────────
flow.load_state_dict(best_state)
flow.eval()
log(f"\n=== Evaluation (best epoch={best_epoch}, val_nll={best_val_nll:.4f}) ===")

# Val NLL baseline (N(0,1))
baseline_nll = 0.5 * float(torch.tensor(y_vl).var().log() + 1 + np.log(2*np.pi))
log(f"  N(0,1) baseline NLL: {baseline_nll:.4f}")

for n_samp in [200, 500]:
    cv_vl = flow_cv(val_r,  C_vl_t, n_samples=n_samp)
    cv_te = flow_cv(test_r, C_te_t, n_samples=n_samp)
    log(f"  ({n_samp} samples)  val_CV={cv_vl:.4f}%  test_CV={cv_te:.4f}%")

log(f"\n  Production flow baseline:  test_CV=1.0494%")
log(f"  HGB kitchen sink:          test_CV=0.8856%")

# ── Save checkpoint ───────────────────────────────────────────────────────────
# Build means/stds dicts keyed by column name for infer_star.py compatibility
# Context is now wider: TAB_COLS + knn_100 + spatial cols
all_ctx_cols = TAB_COLS + ['knn_100'] + list(sp_tr.columns)
means_dict = {col: float(ctx_mean[i]) for i, col in enumerate(all_ctx_cols)}
stds_dict  = {col: float(ctx_std[i])  for i, col in enumerate(all_ctx_cols)}

ckpt = {
    'flow_config': {
        'features': 1,
        'context':  X_tr.shape[1],
        'transforms': TRANSFORMS,
        'hidden_features': HIDDEN,
        'bins': BINS,
    },
    'model_state':    best_state,
    'ctx_mean':       ctx_mean,       # raw array for fast inference
    'ctx_std':        ctx_std,
    'means':          means_dict,     # named dict for infer_star.py compat
    'stds':           stds_dict,
    'y_mean':         y_mean,
    'y_std':          y_std,
    'continuous_cols': all_ctx_cols,  # all context cols (spatial included)
    'cam_cols':       [],             # cam/ccd now part of TAB_COLS as numerics
    'ccd_cols':       [],
    'tab_cols':       TAB_COLS,
    'spatial_cols':   list(sp_tr.columns),
    'alpha':          _alpha,
    'best_val_nll':   best_val_nll,
    'best_epoch':     best_epoch,
    'sec_stats':      sec_stats,      # for sector_mean_loo lookup at inference
    'ref_spatial': ref_full[['tic_id','sector','cam','ccd','col','row','flux_offset_loo']].reset_index(drop=True),
}
torch.save(ckpt, _out_pt)
log(f"\nSaved → {_out_pt}  T={time.time()-t0:.0f}s")
