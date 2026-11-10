"""Frozen-model, local-trend sector continuity diagnostic; no stellar truth claim."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import theilslopes

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stitch.frozen import FrozenPhysicalModel
from restricted_input_hgb import Model

ROOT = Path('eval/baseline_boundary_validation_20261014')
LPV = Path('eval/long_period_variable_expansion_20261014')
CONTROL = Path('eval/pixel_residual_validation_20261013')
METHODS = ['pdcsap', 'full', 'restricted', 'sap', 'header_adjusted_sap']

def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()

def save(p, obj):
    p.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')

def sources():
    rows = []
    for root, sample in [(LPV, 'long_period'), (CONTROL, 'qualified_control_pool')]:
        prod = pd.read_parquet(root/'products.parquet')
        for tic in sorted(prod.tic_id.unique()):
            folder = root/('inference' if root == LPV else '')/str(tic)
            rows.append(dict(tic_id=int(tic), sample=sample, folder=str(folder), root=str(root)))
    return rows

def freeze():
    assert not ROOT.exists(), 'Use a new experiment directory; do not overwrite.'
    rows = sources()
    inputs = {Path(__file__), Path('research_defaults.json'), Path('stitch/frozen.py'),
              Path('eval/restricted_input_hgb.py')}
    for r in rows:
        inputs.add(Path(r['root'])/'products.parquet')
        for name in ['curves.parquet', 'features.parquet', 'predictions.parquet']:
            inputs.add(Path(r['folder'])/name)
    for root in [Path('eval/full_rebuild_20261012/physical'), Path('eval/restricted_input_hgb_20261013/restricted')]:
        inputs.update(root.glob('*.pkl'))
        inputs.add(root/'complete.json')
    ROOT.mkdir()
    save(ROOT/'freeze.json', dict(created_utc=pd.Timestamp.now(tz='UTC').isoformat(), sources=rows,
        inputs={str(p): sha(p) for p in sorted(inputs)},
        question='Do the existing frozen sector factors reduce local-trend discontinuities on real long-period variables?',
        primary_sample='All eight previously examined ASAS-SN long-period variables. Secondary: all 43 existing control-pool sources, some intrinsically variable. All are development diagnostics.',
        model='Frozen original full and restricted HGB. No fitting, reference changes, target centering, or model selection. Gaia RP anchors retained.',
        binning='Common finite positive Q0 archived PDCSAP, SAP and header-adjusted SAP; median 0.25-day bins, >=3 cadences. No per-sector normalization.',
        boundaries='Only sector-number-adjacent pairs with actual accepted-cadence gap 0 < gap <= 3 days. Qualified view requires both individual scores >0.95.',
        estimator='Separate Theil-Sen lines on either side; median(y - slope*t) intercept at common gap midpoint. Primary window 5 days, sensitivities 3 and 7. Require >=8 bins, >=60% of window time span, and nearest accepted bin <=0.5 day from each gap edge.',
        metric='100 times log flux jump right minus left. Per-star median absolute jump; primary summary median of these star scores. Paired full-minus-raw improvement evaluated by median(raw score - model score), 10000 star bootstrap draws, seed 20261014. Positive means smaller discontinuities.',
        pseudo_gaps='Within each participating sector, remove same-duration artificial gap at centers every 2 days, retaining full left/right windows; same estimator. Summarize prediction discrepancies without subtracting them from actual boundary scores. At least one valid pseudo gap on each sector required for primary trend-check view; all eligible boundaries also retained.',
        diagnostics='SAP and header-adjusted SAP are pipeline-stage comparisons, not truth and not HGB inputs. Same-sector pseudo gaps cannot validate sector gains. Synthetic smooth signal with known step must detect and recover step, including gains that cancelled in previous paired injections.',
        limitations='Continuity over <=3-day gaps is a physical assumption checked imperfectly by pseudo gaps. Instrumental edges and stellar curvature confound extrapolation. Does not connect seasons separated by long gaps, certify absolute amplitudes, or measure a noise floor. No independent photometric reference.',
        views=['all', 'qualified'], windows_days=[5., 3., 7.], gap_max_days=3., seed=20261014))
    shutil.copy2(__file__, ROOT/'source_snapshot.py')
    print('Protocol and input hashes frozen', len(rows), 'sources', flush=True)

def fit_edge(d, mid, edge, window, side, column):
    if side == 'left': z = d[(d.time <= edge) & (d.time >= edge-window)]
    else: z = d[(d.time >= edge) & (d.time <= edge+window)]
    z = z[np.isfinite(z[column])]
    if len(z) < 8 or z.time.max()-z.time.min() < .6*window: return None
    near = edge-z.time.max() if side == 'left' else z.time.min()-edge
    if near > .5: return None
    x = z.time.to_numpy()-mid
    y = z[column].to_numpy()
    slope = theilslopes(y, x)[0]
    return float(np.median(y-slope*x))

def jump(a, b, left, right, window, column):
    mid = (left+right)/2
    x = fit_edge(a, mid, left, window, 'left', column)
    y = fit_edge(b, mid, right, window, 'right', column)
    return None if x is None or y is None else 100*(y-x)

def pseudogaps(d, start, end, gap, window):
    out = []
    for mid in np.arange(start+window+gap/2, end-window-gap/2+.00001, 2.):
        j = jump(d, d, mid-gap/2, mid+gap/2, window, 'pdcsap')
        if j is not None: out.append(dict(time=float(mid), jump_log_pp=j))
    return out

def verify_estimator():
    t = np.arange(0., 30., .25)+.125
    y = 8.+.001*t
    a = pd.DataFrame(dict(time=t[t<14], pdcsap=y[t<14]))
    b = pd.DataFrame(dict(time=t[t>16], pdcsap=y[t>16]))
    base = jump(a,b,14.,16.,5.,'pdcsap')
    bad = b.copy(); bad.pdcsap += np.log(1.2/.8)
    shifted = jump(a,bad,14.,16.,5.,'pdcsap')
    fixed = bad.copy(); fixed.pdcsap -= np.log(1.2/.8)
    recovered = jump(a,fixed,14.,16.,5.,'pdcsap')
    np.testing.assert_allclose([base, recovered], [0,0], atol=1e-10)
    np.testing.assert_allclose(shifted,100*np.log(1.5),atol=1e-10)
    # Adding a common flux unit conversion cannot change the boundary score.
    aa=a.copy();bb=b.copy();aa.pdcsap+=3;bb.pdcsap+=3
    np.testing.assert_allclose(jump(aa,bb,14.,16.,5.,'pdcsap'),base,atol=1e-10)
    return dict(smooth_log_linear_jump=base, wrong_fixed_sector_gain_jump=shifted,
                corrected_jump=recovered, expected_wrong_jump=100*np.log(1.5))

def run():
    cfg = json.loads((ROOT/'freeze.json').read_text())
    assert not (ROOT/'boundaries.csv').exists()
    for p,h in cfg['inputs'].items(): assert sha(p)==h, p
    checks = verify_estimator()
    full = FrozenPhysicalModel(); restricted = Model('restricted')
    edges=[]; pseudos=[]; attr=[]; bins=[]; sectors=[]; replay=[]
    for r in cfg['sources']:
        tic=r['tic_id']; folder=Path(r['folder'])
        f=pd.read_parquet(folder/'features.parquet')
        p=pd.read_parquet(folder/'predictions.parquet').set_index('sector')
        c=pd.read_parquet(folder/'curves.parquet')
        prod=pd.read_parquet(Path(r['root'])/'products.parquet').set_index(['tic_id','sector'])
        full_factor=full.predict(f); rest_factor=np.exp(restricted.predict_log(f))
        np.testing.assert_allclose(full_factor, p.loc[f.sector,'correction'],rtol=1e-10)
        assert f.has_gaia_anchor.all() and p.reference_supported.all()
        replay.append(dict(tic_id=tic, sectors=len(f), max_prediction_difference=float(np.max(abs(full_factor-p.loc[f.sector,'correction'].to_numpy())))))
        fs=dict(zip(f.sector,full_factor)); rs=dict(zip(f.sector,rest_factor))
        bysector={}; metadata={}
        for sector,z in c.groupby('sector'):
            meta=prod.loc[(tic,sector)]
            crowd=float(meta.crowdsap) if 'crowdsap' in meta else float(f.set_index('sector').loc[sector,'crowdsap'])
            capture=float(meta.flfrcsap) if 'flfrcsap' in meta else float(f.set_index('sector').loc[sector,'flfrcsap'])
            valid=np.isfinite(z.time)&np.isfinite(z.raw)&(z.raw>0)&np.isfinite(z.sap)&(z.sap>0)
            adjusted=(z.sap-(1-crowd)*np.median(z.loc[valid,'sap']))/capture
            valid &= np.isfinite(adjusted)&(adjusted>0)
            q=z.loc[valid].copy()
            q['pdcsap']=np.log(q.raw);q['sap']=np.log(q.sap)
            q['header_adjusted_sap']=np.log(adjusted.loc[valid])
            q['full']=q.pdcsap-np.log(fs[sector]);q['restricted']=q.pdcsap-np.log(rs[sector])
            q['bin']=np.floor(q.time/.25).astype(int)
            g=q.groupby('bin'); b=g[['time']+METHODS].median()
            b['cadences']=g.size(); b=b[b.cadences>=3].reset_index(drop=True)
            qualified=bool(p.loc[sector,'systematic_score']>.95)
            m=dict(tic_id=tic,sample=r['sample'],sector=int(sector),qualified=qualified,
                   tars_unknown=bool(pd.isna(p.loc[sector,'systematic_score'])),
                   original_cadences=len(z),common_cadences=len(q),retained_binned_cadences=int(b.cadences.sum()), bins=len(b),
                   start=float(q.time.min()),end=float(q.time.max()),full_factor=fs[sector],restricted_factor=rs[sector])
            metadata[sector]=m;bysector[sector]=b;sectors.append(m)
            bins.append(b.assign(tic_id=tic,sector=sector,sample=r['sample']))
        seq=sorted(metadata, key=lambda s: metadata[s]['start'])
        for sa,sb in zip(seq[:-1],seq[1:]):
            a=metadata[sa];b=metadata[sb];gap=b['start']-a['end']
            pair=dict(tic_id=tic,sample=r['sample'],sector_left=int(sa),sector_right=int(sb),gap_days=gap,
                      qualified=a['qualified'] and b['qualified'],tars_unknown=a['tars_unknown'] or b['tars_unknown'])
            reason='eligible' if sb==sa+1 and 0<gap<=3 else ('nonadjacent_sectors' if sb!=sa+1 else 'gap_outside_0_3_days')
            attr.append(dict(**pair,status=reason))
            if reason!='eligible':continue
            for window in cfg['windows_days']:
                raw=jump(bysector[sa],bysector[sb],a['end'],b['start'],window,'pdcsap')
                if raw is None:
                    edges.append(dict(**pair,window_days=window,available=False,trend_checked=False));continue
                pa=pseudogaps(bysector[sa],a['start'],a['end'],gap,window)
                pb=pseudogaps(bysector[sb],b['start'],b['end'],gap,window)
                row=dict(**pair,window_days=window,available=True,trend_checked=bool(pa and pb),
                         pseudo_left=len(pa),pseudo_right=len(pb),
                         pseudo_median_abs_log_pp=float(np.median([abs(v['jump_log_pp']) for v in pa+pb])) if pa or pb else np.nan)
                for method in METHODS:
                    row[method]=jump(bysector[sa],bysector[sb],a['end'],b['start'],window,method)
                for method,key in [('full','full_factor'),('restricted','restricted_factor')]:
                    expected=raw-100*np.log(b[key]/a[key])
                    np.testing.assert_allclose(row[method],expected,atol=2e-10)
                edges.append(row)
                for sec,ps in [(sa,pa),(sb,pb)]:
                    pseudos.extend(dict(**pair,window_days=window,pseudo_sector=int(sec),**v) for v in ps)
        print('Scored',r['sample'],tic,len(metadata),'sectors',flush=True)
    e=pd.DataFrame(edges);e.to_csv(ROOT/'boundaries.csv',index=False)
    pd.DataFrame(pseudos).to_parquet(ROOT/'pseudo_gaps.parquet',index=False)
    pd.DataFrame(attr).to_csv(ROOT/'pair_attrition.csv',index=False)
    pd.DataFrame(sectors).to_csv(ROOT/'sector_attrition.csv',index=False)
    pd.concat(bins,ignore_index=True).to_parquet(ROOT/'binned_curves.parquet',index=False)
    pd.DataFrame(replay).to_csv(ROOT/'prediction_replay.csv',index=False)
    summarize(e)
    for p,h in cfg['inputs'].items(): assert sha(p)==h, p
    checks.update(input_hashes_unchanged=len(cfg['inputs']),full_inventory_verified=len(full.hashes),
                  sources=len(cfg['sources']),sectors=len(sectors),missing_anchor_fallbacks=0,
                  unsupported_references=0,exact_factor_jump_identity=True,model_training=False)
    save(ROOT/'verification.json',checks)

def summarize(e):
    rows=[];individual=[];rng=np.random.default_rng(20261014)
    for (sample,window),z in e.groupby(['sample','window_days']):
        for view in ['all','qualified']:
            q=z if view=='all' else z[z.qualified]
            for eligibility in ['all_eligible','trend_checked']:
                q1=q[q.available & (q.trend_checked if eligibility=='trend_checked' else True)]
                meta=dict(sample=sample,window_days=window,view=view,eligibility=eligibility,boundaries=len(q1),stars=q1.tic_id.nunique())
                if not len(q1):rows.append(dict(**meta,method='unavailable'));continue
                star=q1.groupby('tic_id')[METHODS].agg(lambda x: np.median(np.abs(x)))
                pseudo=q1.groupby('tic_id').pseudo_median_abs_log_pp.median()
                for method in METHODS:
                    delta=star.pdcsap-star[method]
                    draw=rng.integers(0,len(star),(10000,len(star)))
                    ci=np.percentile(np.median(delta.to_numpy()[draw],axis=1),[2.5,97.5])
                    rows.append(dict(**meta,method=method,median_star_abs_jump_log_pp=float(star[method].median()),
                                     median_paired_improvement_log_pp=float(delta.median()),
                                     improvement_ci_low=ci[0],improvement_ci_high=ci[1],stars_improved=int((delta>1e-10).sum()),
                                     median_star_pseudo_gap_error_log_pp=float(pseudo.median())))
                    for tic,value in star[method].items():
                        individual.append(dict(**meta,tic_id=int(tic),method=method,abs_jump_log_pp=value,
                            source_boundaries=int(q1.tic_id.eq(tic).sum()),improvement_log_pp=float(delta.loc[tic]),pseudo_error_log_pp=float(pseudo.loc[tic])))
    pd.DataFrame(rows).to_csv(ROOT/'summary.csv',index=False)
    pd.DataFrame(individual).to_csv(ROOT/'per_star.csv',index=False)
    print(pd.DataFrame(rows).query("window_days==5 and eligibility=='trend_checked'").to_string(index=False),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['freeze','run'])
    args=parser.parse_args()
    freeze() if args.action=='freeze' else run()
