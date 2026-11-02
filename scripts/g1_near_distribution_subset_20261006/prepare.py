"""Lock feature-only subsets before training or reading any new forecast errors."""
from pathlib import Path
import sys, json, shutil, hashlib
from collections import Counter
import numpy as np
from scipy.optimize import milp, Bounds, LinearConstraint

ROOT=Path(__file__).resolve().parents[2]
VERSION='g1_near_distribution_subset_20261006'
OUT=ROOT/'outputs'/VERSION
BANK=ROOT/'outputs/transformer_matched97_20261006/new'
SURVEY=ROOT/'outputs/g1_population_survey_20261006'
ORDER='EDGC'
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),'utf8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def weighted_quantile(x,w,q):
    ids=np.argsort(x,kind='stable');return x[ids[np.searchsorted(np.cumsum(w[ids])/sum(w),q)]]
def design_matrix(x,ref,w,thresholds):
    rows=[];target=[];cost=[];labels=[]
    def add(a,b,c,label):rows.append(a);target.append(float(w@b));cost.append(c);labels.append(label)
    for j,k in enumerate(ORDER):
        cuts=sorted(set([weighted_quantile(ref[:,j],w,q) for q in [.25,.5,.75]]+[thresholds[k]]))
        for c in cuts:add(x[:,j]<=c,ref[:,j]<=c,1/len(cuts),f'{k}_cdf_{c:.12g}')
    bins=np.digitize(x[:,3],[.2,1,1.6]);rbins=np.digitize(ref[:,3],[.2,1,1.6])
    for b in range(4):add(bins==b,rbins==b,3/4,f'C_speed_bin_{b}')
    levels=x>=np.array([thresholds[k] for k in ORDER]);rl=ref>=np.array([thresholds[k] for k in ORDER])
    for code in range(16):
        bits=np.array(list(map(int,f'{code:04b}')))
        add(np.all(levels==bits,axis=1),np.all(rl==bits,axis=1),1/16,f'EDGC_{code:04b}')
    return np.array(rows,float),np.array(target),np.array(cost),labels
def choose(indices,meta,x,ref,w,thresholds,n,seed):
    A,t,c,labels=design_matrix(x,ref,w,thresholds);m,k=A.shape
    recs=sorted({meta[i]['recording'] for i in indices})
    recmatrix=np.array([[meta[i]['recording']==r for i in indices] for r in recs],float)
    # L1 matching of marginal CDFs, camera-speed categories and joint EDGC cells.
    # 70% retention, min one per available recording, max 40% in one recording.
    eq=np.c_[A/n,-np.eye(m),np.eye(m)]
    total=np.r_[np.ones(k),np.zeros(2*m)][None]
    rec=np.c_[recmatrix,np.zeros((len(recs),2*m))]
    rng=np.random.default_rng(seed)
    objective=np.r_[rng.uniform(0,1e-8,k),c,c]
    constraints=[LinearConstraint(eq,t,t),LinearConstraint(total,n,n),LinearConstraint(rec,1,max(1,int(np.ceil(.4*n))))]
    result=milp(objective,integrality=np.r_[np.ones(k),np.zeros(2*m)],bounds=Bounds(np.zeros(k+2*m),np.r_[np.ones(k),np.full(2*m,np.inf)]),constraints=constraints,options={'time_limit':25,'mip_rel_gap':.002})
    assert result.x is not None,result.message
    keep=np.flatnonzero(result.x[:k]>.5);assert len(keep)==n
    count=recmatrix[:,keep].sum(1);assert min(count)>=1 and max(count)<=np.ceil(.4*n)
    return sorted(indices[j] for j in keep),dict(status=int(result.status),message=result.message,gap=float(result.mip_gap),labels=labels,target=t.tolist(),cost=c.tolist(),pool=(A.mean(1)).tolist(),matched=(A[:,keep].mean(1)).tolist(),pool_distance=float(c@abs(A.mean(1)-t)),matched_distance=float(c@abs(A[:,keep].mean(1)-t)))
def main():
    assert not (OUT/'selection_locked.json').exists(),'Selection already frozen'
    meta=read(BANK/'data/samples.json');data=dict(np.load(BANK/'data/dataset.npz'));folds=read(BANK/'folds.json')
    desc=read(ROOT/'outputs/g1_edgc_training_distribution_20261006/window_distribution.json')
    survey=[r for r in read(SURVEY/'measured_features.json') if r['near']]
    thresholds=read(SURVEY/'protocol_locked.json')['thresholds']
    assert all(a['key']==b['key'] for a,b in zip(meta,desc))
    eligible=[i for i,r in enumerate(desc) if r['prior_inventory_eligible'] and r['expected_sampling_probability']>0 and np.min((data['boxes'][i,:,3]-data['boxes'][i,:,1])/1920)>=.08]
    assert len(eligible)==183
    spec=dict(name='Near G1 empirical distribution subset trial',seed=2026100621,retention=.70,
      source='Only previously used matched97_new training windows; no new labels',eligible=eligible,
      selection='Each fold independently selects 70% (floor) of existing near/G1 eligible training windows. Weighted L1 match of historical E,D,G,C marginal quartile/old-threshold CDFs (weight 1 each), four C speed bins (weight 3 total), and 16 EDGC binary joint cells (weight 1 total). One or more windows from every available training recording; maximum 40% per recording.',
      target='Probability-weighted survey near people; exclude the fold test AND validation recordings before computing target CDFs. Survey labels never become forecasting labels.',
      random_control='Same per-fold sample count from same eligible pool, with one uniform random window per recording then SRS without replacement over remaining windows; no error or feature matching.',
      sampler='Both new arms uniformly sample subset windows with replacement. Existing baseline used uniform recording then window; baseline comparison changes eligibility, count, composition and sampler jointly. Matched versus random holds count/eligibility/sampler fixed.',
      unsupported='Keep original empirical target; unsupported C categories incur irreducible matching residual. Do not relabel, fabricate or claim exact matching.',
      primary='All six fixed architectures, seeds 17/29/43, exact old199 test/validation indices, 120 epochs x2 updates, 48 batch. Fixed matching held-out cameras. Report old199 plus prespecified strict-near G1 test subset (87) and per-recording/seed/sparse checks.',
      interpretation='One locked subset draw per arm; stochastic training seeds are not independent subset draws. Approximate distribution match and exploratory policy comparison; not proof of causal factor importance.')
    write(OUT/'design_locked.json',spec)
    arms={a:[] for a in ['matched','random']};audit=[]
    for fold in folds:
        indices=[i for i in fold['train_indices'] if i in eligible];n=int(np.floor(.7*len(indices)))
        rr=[r for r in survey if r['recording'] not in [fold['test'],fold['validation']]]
        ref=np.array([[r['features'][k] for k in ORDER] for r in rr]);w=np.array([r['survey_weight'] for r in rr]);w/=sum(w)
        x=np.array([[desc[i]['features'][k] for k in ORDER] for i in indices])
        selected,diag=choose(indices,meta,x,ref,w,thresholds,n,spec['seed']+fold['fold'])
        rng=np.random.default_rng(spec['seed']+1000+fold['fold']);recs=sorted({meta[i]['recording'] for i in indices})
        random=[int(rng.choice([i for i in indices if meta[i]['recording']==r])) for r in recs]
        random=sorted(random+rng.choice(sorted(set(indices)-set(random)),n-len(random),replace=False).tolist())
        A,t,c,_=design_matrix(x,ref,w,thresholds);pos=[indices.index(i) for i in random]
        diag.update(random=(A[:,pos].mean(1)).tolist(),random_distance=float(c@abs(A[:,pos].mean(1)-t)))
        for arm,keep in [('matched',selected),('random',random)]:
            assert set(keep)<=set(fold['train_indices']) and len(keep)==len(set(keep))==n
            assert all(meta[i]['recording'] not in [fold['test'],fold['validation']] for i in keep)
            arms[arm].append(dict(fold,train_indices=keep,train=sorted({meta[i]['recording'] for i in keep}),reuse_old_checkpoint=False,subset_arm=arm,eligible_train_indices=indices))
        audit.append(dict(fold=fold['fold'],available=len(indices),selected=n,target_near_n=len(rr),excluded_target_recordings=[fold['test'],fold['validation']],target_keys=[r['key'] for r in rr],matched_keys=[meta[i]['key'] for i in selected],random_keys=[meta[i]['key'] for i in random],diagnostics=diag))
        print(f'fold {fold["fold"]}: {len(indices)} -> {n}; distance pool {diag["pool_distance"]:.3f} matched {diag["matched_distance"]:.3f} random {diag["random_distance"]:.3f}',flush=True)
    cfg=read(BANK/'protocol.json')
    for arm in arms:
        dest=OUT/arm;write(dest/'folds.json',arms[arm]);write(dest/'protocol.json',dict(cfg,name=spec['name']+' '+arm,approved='User requested near G1 distribution subset retraining',plan='outputs/'+VERSION+'/design_locked.json',sampler=spec['sampler'],cohort='Subset of 183 near/G1 eligible existing training windows, 70% per-fold retention',test_rule=spec['primary']))
    write(OUT/'selection_locked.json',dict(arms=['matched','random'],folds=audit,design_sha256=sha(OUT/'design_locked.json'),eligible_windows=len(eligible),test_near_G1_indices=[i for i in range(199) if i in eligible]))
    protected=list(BANK.rglob('*.pt'))+list(BANK.glob('data/*'))+[BANK/'folds.json',BANK/'protocol.json',BANK/'metrics.json',BANK/'summary.json',SURVEY/'measured_features.json',SURVEY/'protocol_locked.json',ROOT/'outputs/g1_edgc_training_distribution_20261006/window_distribution.json']
    protected+=list((ROOT/'scripts/transformer_matched97_20261006').glob('*.py'))
    write(OUT/'protected_previous_hashes.json',{p.relative_to(ROOT).as_posix():sha(p) for p in protected})
    paths=[OUT/'design_locked.json',OUT/'selection_locked.json']+[OUT/a/f for a in arms for f in ['folds.json','protocol.json']]
    write(OUT/'selection_hashes.json',{p.relative_to(OUT).as_posix():sha(p) for p in paths})
if __name__=='__main__':main()
