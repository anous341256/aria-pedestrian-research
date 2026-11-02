"""Freeze the 199-window cohort, 11 held-out pairs and unchanged budgets."""
from shared import *
import copy,collections

def main():
 assert not (OUT/'protocol.json').exists();DATA.mkdir(parents=True,exist_ok=True)
 extra=ROOT/'outputs/sam2_g1_expansion_20261005/dataset';old=dict(np.load(OLD/'dataset.npz'));new=dict(np.load(extra/'clear_dataset.npz'));oldmeta=read(OLD/'samples.json');newmeta=read(extra/'clear_samples.json')
 assert len(oldmeta)==117 and len(newmeta)==82
 raw=[];rawvalid=[];cache={}
 for j,m in enumerate(newmeta):
  cid=m['clip']
  if cid not in cache:cache[cid]={(r['id'],r['frame']):r for r in gz(ROOT/f'outputs/sam2_g1_expansion_20261005/{cid}/masks_rle.jsonl.gz')}
  points=[];valid=[]
  for f in m['frames'][:P]:
   row=cache[cid].get((m['id'],f));points.append((np.array(row['box'][:2])+row['box'][2:])/2 if row else [0.,0.]);valid.append(1. if row else 0.)
  if valid[-1]==0:points[-1]=new['xy'][j,P-1];valid[-1]=1.
  raw.append(points);rawvalid.append(valid)
 new.update(raw=np.array(raw,np.float32),raw_valid=np.array(rawvalid,np.float32))
 data={k:np.concatenate([old[k],new[k]]) for k in new};np.savez_compressed(DATA/'dataset.npz',**data)
 meta=[]
 for version,items in [('old117',oldmeta),('new82',newmeta)]:
  for m in items:
   i=len(meta);b=data['boxes'][i];edge=bool(((b[:,0]<=5)|(b[:,1]<=5)|(b[:,2]>=2555)|(b[:,3]>=1915)).any())
   source=m.copy();source.update(source_version=version,edge=edge,static_control=(m['clip']=='19'),classification=m.get('classification','previous_reviewed_cohort'))
   if version=='new82':source['source_start_s']=read(ROOT/f'outputs/sam2_g1_expansion_20261005/{m["clip"]}/input/metadata.json')['source']['start_s']
   meta.append(source)
 write(DATA/'samples.json',meta);assert len({m['key'] for m in meta})==199
 groups=sorted({m['recording'] for m in meta});assert len(groups)==11
 oldfolds=read(OLD/'folds.json');oldcamera=read(ORIGINAL_CAMERA/'folds.json');pairs=[(f['test'],f['validation']) for f in oldfolds]
 for g in groups:
  if g not in [x[0] for x in pairs]:pairs.append((g,groups[(groups.index(g)+1)%len(groups)]))
 camera_data=np.load(ORIGINAL_CAMERA/'dataset.npz');rec=camera_data['record_index'];recordnames=read(ORIGINAL_CAMERA/'inventory.json')['record_names'];folds=[];cfolds=[]
 for f,(test,val) in enumerate(pairs):
  train=[g for g in groups if g not in [test,val]]
  fold=dict(fold=f,test=test,validation=val,train=train,train_indices=[i for i,m in enumerate(meta) if m['recording'] in train],val_indices=[i for i,m in enumerate(meta) if m['recording']==val],test_indices=[i for i,m in enumerate(meta) if m['recording']==test])
  assert fold['train_indices'] and fold['val_indices'] and fold['test_indices'];folds.append(fold)
  ct=[g for g in recordnames if g not in [test,val]]
  cf=dict(fold=f,test=test,validation=val,train=ct,train_indices=np.flatnonzero(np.isin(rec,[recordnames.index(g) for g in ct])).tolist(),val_indices=np.flatnonzero(rec==recordnames.index(val)).tolist(),test_indices=np.flatnonzero(rec==recordnames.index(test)).tolist(),reuse_original_fold=f if f<6 else None)
  if f<6:
   assert test==oldcamera[f]['test'] and val==oldcamera[f]['validation']
   assert cf['train_indices']==oldcamera[f]['train_indices'] and cf['val_indices']==oldcamera[f]['val_indices']
  cfolds.append(cf)
 assert sorted(i for f in folds for i in f['test_indices'])==list(range(199))
 write(OUT/'folds.json',folds);write(CAMERA/'folds.json',cfolds)
 cfg=read(BASELINE/'protocol.json');cfg.update(name='Expanded-data A–F rerun',approved='User asked to run models on expanded dataset 2026-10-05',cohort='117 original + 82 strict new; 199 windows; 11 recordings',models=198,camera='Reuse matching six old exclusion pairs (18 checkpoints), train five new held-out pairs (15 checkpoints). No all-recording refit.',plan='plans/TRANSFORMER_EXPANDED_20261005.md',old_test_comparison='Original six test/validation recording pairs and camera models preserved. New validation windows can alter selected checkpoint.',training_augmentation='Same 40/20/40 reviewed/pre-review/noise mixture; pre-review masks for new windows drawn from preserved raw same-ID archive. Final historical point always observed.')
 write(OUT/'protocol.json',cfg);ccfg=read(ORIGINAL_CAMERA/'protocol.json');ccfg.update(split='11 recording-disjoint pairs; five newly trained, six exact old pairs reused',full_refit='None authorized or run in this experiment');write(CAMERA/'protocol.json',ccfg)
 # Fixed new visualization selection before training/test scores.
 fixed=[]
 for cid in sorted({m['clip'] for m in newmeta}):
  mm=[m for m in newmeta if m['clip']==cid];identity=sorted({m['id'] for m in mm})[0];mm=sorted([m for m in mm if m['id']==identity],key=lambda m:m['cutoff']);fixed.append(mm[len(mm)//2]['key'])
 write(OUT/'fixed_cases.json',dict(old=read(OLD/'presentation/selection.json')['fixed'],new=fixed,rule='First eligible identity per clip, median cutoff, frozen before fitting'))
 paths=set(ROOT/p for p in read(BASELINE/'source_hashes.json'))
 paths.update(p for p in BASELINE.rglob('*') if p.is_file())
 paths.update(p for p in (ROOT/'scripts/transformer_fusion_20261005').glob('*.py'))
 paths.update(p for p in extra.glob('*') if p.is_file())
 for cid in sorted({m['clip'] for m in newmeta}):
  paths.update(p for p in (ROOT/f'outputs/sam2_g1_expansion_20261005/{cid}/final').glob('*') if p.is_file())
  paths.add(ROOT/f'outputs/sam2_g1_expansion_20261005/{cid}/masks_rle.jsonl.gz')
 paths.update([ORIGINAL_CAMERA/'dataset.npz',ORIGINAL_CAMERA/'inventory.json',ROOT/'plans/TRANSFORMER_EXPANDED_20261005.md'])
 write(OUT/'source_hashes.json',{p.relative_to(ROOT).as_posix():sha(p) for p in sorted(paths) if p.is_file()})
 write(DATA/'selection.json',dict(samples=199,old=117,new=82,recordings=11,static_new=3,clips=dict(collections.Counter(m['clip'] for m in meta)),fold_counts=[dict(fold=f['fold'],train=len(f['train_indices']),validation=len(f['val_indices']),test=len(f['test_indices'])) for f in folds],old_arrays_exact={k:bool(np.array_equal(data[k][:117],old[k])) for k in new}))
 print('Frozen 199 windows; folds',[(len(f['train_indices']),len(f['val_indices']),len(f['test_indices'])) for f in folds],flush=True)
if __name__=='__main__':main()
