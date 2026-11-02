"""Freeze cohort, exact previous validation windows and matching camera pairs."""
from shared import *
import collections,shutil

def main():
 assert not (OUT/'protocol.json').exists();DATA.mkdir(parents=True,exist_ok=True)
 source=EXPANSION/'dataset';old=dict(np.load(BASELINE/'data/dataset.npz'));oldmeta=read(BASELINE/'data/samples.json')
 data=dict(np.load(source/'combined_strict_dataset.npz'));meta=read(source/'combined_strict_samples.json')
 assert len(meta)==296 and len(oldmeta)==199 and meta[:199]==oldmeta
 exact={k:bool(np.array_equal(data[k][:199],old[k])) for k in old};assert all(exact.values())
 assert len({m['key'] for m in meta})==296
 shutil.copy2(source/'combined_strict_dataset.npz',DATA/'dataset.npz');write(DATA/'samples.json',meta)
 groups=sorted({m['recording'] for m in meta});assert len(groups)==13
 oldfolds=read(BASELINE/'folds.json');oldcamera=read(BASELINE/'camera/folds.json')
 pairs=[(f['test'],f['validation']) for f in oldfolds]
 for g in groups:
  if g not in [p[0] for p in pairs]:pairs.append((g,groups[(groups.index(g)+1)%len(groups)]))
 camera_data=np.load(ORIGINAL_CAMERA/'dataset.npz');rec=camera_data['record_index'];recordnames=read(ORIGINAL_CAMERA/'inventory.json')['record_names'];folds=[];cfolds=[]
 for f,(test,val) in enumerate(pairs):
  train=[g for g in groups if g not in [test,val]]
  val_all=[i for i,m in enumerate(meta) if m['recording']==val]
  val_used=oldfolds[f]['val_indices'] if f<11 else val_all
  fold=dict(fold=f,test=test,validation=val,train=train,train_indices=[i for i,m in enumerate(meta) if m['recording'] in train],val_indices=val_used,test_indices=[i for i,m in enumerate(meta) if m['recording']==test],unused_validation_indices=sorted(set(val_all)-set(val_used)),validation_rule='Exact previous indices' if f<11 else 'All valid windows of validation recording')
  assert fold['train_indices'] and fold['val_indices'] and fold['test_indices']
  assert sorted(fold['train_indices']+fold['val_indices']+fold['test_indices']+fold['unused_validation_indices'])==list(range(296))
  if f<11:
   assert [i for i in fold['test_indices'] if i<199]==oldfolds[f]['test_indices']
   assert [i for i in fold['train_indices'] if i<199]==oldfolds[f]['train_indices']
  folds.append(fold)
  ct=[g for g in recordnames if g not in [test,val]]
  cf=dict(fold=f,test=test,validation=val,train=ct,train_indices=np.flatnonzero(np.isin(rec,[recordnames.index(g) for g in ct])).tolist(),val_indices=np.flatnonzero(rec==recordnames.index(val)).tolist(),test_indices=np.flatnonzero(rec==recordnames.index(test)).tolist(),reuse_previous_fold=f if f<11 else None)
  if f<11:
   assert all(cf[k]==oldcamera[f][k] for k in ['test','validation','train','train_indices','val_indices','test_indices'])
  cfolds.append(cf)
 assert sorted(i for f in folds for i in f['test_indices'])==list(range(296))
 write(OUT/'folds.json',folds);write(CAMERA/'folds.json',cfolds)
 cfg=read(BASELINE/'protocol.json');cfg.update(name='Second expanded-data A–F rerun',approved='User requested adding second expansion and evaluating',cohort='117 original + 82 first + 97 second; 296 windows; 13 recordings',models=234,camera='Reuse 33 matching old cameras; train six cameras for two new exclusion pairs. No all-recording refit.',plan='plans/TRANSFORMER_EXPANDED_V2_20261005.md',old_test_comparison='Previous eleven test/validation recording pairs, exact validation window indices and camera checkpoints frozen. Only eligible training windows added.',start_date='2026-10-05')
 write(OUT/'protocol.json',cfg);ccfg=read(BASELINE/'camera/protocol.json');ccfg.update(split='13 recording-disjoint exclusion pairs; eleven reused and two new',full_refit='None');write(CAMERA/'protocol.json',ccfg)
 fixed=[]
 for cid in sorted({m['clip'] for m in meta[199:]}):
  mm=[m for m in meta[199:] if m['clip']==cid];identity=sorted({m['id'] for m in mm})[0];mm=sorted([m for m in mm if m['id']==identity],key=lambda m:m['cutoff']);fixed.append(mm[len(mm)//2]['key'])
 oldcases=read(BASELINE/'fixed_cases.json');write(OUT/'fixed_cases.json',dict(old=oldcases['old'],first=oldcases['new'],new=fixed,rule='Previous 18 cases preserved; first eligible identity and median cutoff per new clip, before fitting'))
 paths={ROOT/p for p in read(BASELINE/'source_hashes.json')}
 paths.update(p for p in BASELINE.rglob('*') if p.is_file())
 paths.update(p for p in (ROOT/'scripts/transformer_expanded_20261005').glob('*.py'))
 paths.update(p for p in (ROOT/'reports/transformer_expanded_20261005').glob('*') if p.is_file())
 paths.update(p for p in source.glob('*') if p.is_file())
 for cid in sorted({m['clip'] for m in meta[199:]}):
  paths.update(p for p in (EXPANSION/cid/'final').glob('*') if p.is_file());paths.add(EXPANSION/cid/'masks_rle.jsonl.gz')
 paths.update([ORIGINAL_CAMERA/'dataset.npz',ORIGINAL_CAMERA/'inventory.json',ROOT/'plans/TRANSFORMER_EXPANDED_V2_20261005.md'])
 write(OUT/'source_hashes.json',{p.relative_to(ROOT).as_posix():sha(p) for p in sorted(paths) if p.is_file()})
 write(DATA/'selection.json',dict(samples=296,old=199,new=97,recordings=13,static_new=10,static_all=13,clips=dict(collections.Counter(m['clip'] for m in meta)),fold_counts=[dict(fold=f['fold'],train=len(f['train_indices']),validation=len(f['val_indices']),test=len(f['test_indices']),unused_validation=len(f['unused_validation_indices'])) for f in folds],old_arrays_exact=exact,old_metadata_exact=True,validation_windows_frozen=True))
 print('Frozen 296 windows; 13 folds; previous validation windows exact.',flush=True)
if __name__=='__main__':main()
