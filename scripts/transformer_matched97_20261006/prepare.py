"""Freeze two equal-size annotation pools and paired count/static-matched folds."""
from shared import *
import collections,shutil,json
PRIOR=ROOT/'outputs/transformer_expanded_v2_20261005'
def spread(indices,meta,n):
 indices=sorted(indices,key=lambda i:(meta[i]['recording'],meta[i]['clip'],meta[i]['id'],meta[i]['cutoff']))
 return [indices[j] for j in np.linspace(0,len(indices)-1,n,dtype=int)] if n else []
def main():
 assert not (VERSION/'selection_locked.json').exists()
 oldmeta=read(BASELINE/'data/samples.json');base=dict(np.load(BASELINE/'data/dataset.npz'));assert len(oldmeta)==199
 datasets={'old':dict(np.load(PRIOR/'data/dataset.npz')),'new':dict(np.load(EXPANSION/'dataset/combined_strict_dataset.npz'))}
 metas={'old':read(PRIOR/'data/samples.json'),'new':read(EXPANSION/'dataset/combined_strict_samples.json')}
 for arm in metas:
  assert len(metas[arm])==296 and metas[arm][:199]==oldmeta
  assert all(np.array_equal(datasets[arm][k][:199],base[k]) for k in base)
 folds0=read(BASELINE/'folds.json');oldfolds=read(PRIOR/'folds.json');cfg=read(BASELINE/'protocol.json');foldsets={'old':[],'new':[]};audit=[]
 for f in folds0:
  pair={f['test'],f['validation']};strata={}
  for arm in metas:
   strata[arm]={str(s):[i for i,m in enumerate(metas[arm]) if i>=199 and m['recording'] not in pair and bool(m.get('static_control',False))==s] for s in [False,True]}
  counts={s:min(len(strata['old'][s]),len(strata['new'][s])) for s in ['False','True']}
  for arm in metas:
   adds=sorted(sum([spread(strata[arm][s],metas[arm],counts[s]) for s in counts],[]));train=sorted(f['train_indices']+adds)
   omitted=sorted(set(range(199,296))-set(adds))
   fold=dict(f,train_indices=train,test_indices=f['test_indices'],unused_added_indices=omitted,added_train_indices=adds,
             moving_added=counts['False'],static_added=counts['True'],train=sorted({metas[arm][i]['recording'] for i in train}),
             reuse_old_checkpoint=arm=='old' and train==oldfolds[f['fold']]['train_indices'])
   assert sorted(train+fold['val_indices']+fold['test_indices']+omitted)==list(range(296))
   foldsets[arm].append(fold)
  audit.append(dict(fold=f['fold'],test=f['test'],validation=f['validation'],available={a:{s:len(v) for s,v in z.items()} for a,z in strata.items()},matched_counts=counts,
                    base_train=len(f['train_indices']),reuse_old=foldsets['old'][-1]['reuse_old_checkpoint']))
 for arm in metas:
  dest=VERSION/arm;(dest/'data').mkdir(parents=True,exist_ok=True);np.savez_compressed(dest/'data/dataset.npz',**datasets[arm]);write(dest/'data/samples.json',metas[arm]);write(dest/'folds.json',foldsets[arm])
  conf=dict(cfg,name='Equal-size batch replacement: '+arm,models=198,cohort='199 fixed old windows plus a 97-window added pool; per-fold moving/static counts matched',
    approved='User requested equal-size newly annotated G1 comparison',test_rule='Fixed 199 original tests only; no new labels in headline evaluation',plan='plans/G1_MATCHED_BATCH_20261006.md',start_date='2026-10-06')
  conf.update(camera='Exactly reuse all 33 matching held-out camera checkpoints from the 199-window experiment. No camera training.',old_test_comparison='Exact old199 test and validation indices; only added training pool varies.')
  write(dest/'protocol.json',conf);write(dest/'camera/folds.json',read(BASELINE/'camera/folds.json'));write(dest/'camera/protocol.json',read(BASELINE/'camera/protocol.json'))
 write(VERSION/'selection_locked.json',dict(arms=['old','new'],base=199,pool_size=97,folds=audit,selection='Separate static/moving strata; deterministic temporal spread before any forecast scores',new_selected_sha256=sha(EXPANSION/'dataset/clear_samples.json'),source_mismatch='Recording and motion distributions remain different; comparison is not label-quality causal identification'))
 paths=[p for p in PRIOR.rglob('*') if p.is_file()]+[p for p in (ROOT/'scripts/transformer_expanded_v2_20261005').glob('*.py')]
 write(VERSION/'protected_previous_hashes.json',{p.relative_to(ROOT).as_posix():sha(p) for p in paths})
 print(json.dumps(audit,indent=2),flush=True)
if __name__=='__main__':main()
