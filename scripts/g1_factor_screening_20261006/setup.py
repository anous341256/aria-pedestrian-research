"""Prepare the locked scenario evaluation without training or changing weights."""
from pathlib import Path
import json,hashlib,collections
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/g1_factor_screening_20261006'
DESIGN=ROOT/'outputs/g1_factor_redesign_20261006'
BANK=ROOT/'outputs/transformer_matched97_20261006/new'
VARIANTS=['A_early_fusion','B_person_only','C_separated','D_bidirectional','E_oneway','F_interacting_geometry']
SEEDS=[17,29,43]
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
 OUT.mkdir(parents=True,exist_ok=True);assert not (OUT/'prepared.json').exists()
 # Hash the frozen design, source data and every checkpoint before evaluation.
 protected={}
 manifest=read(DESIGN/'manifest.json')
 for f,h in manifest['files'].items():assert sha(DESIGN/f)==h
 for p in DESIGN.iterdir():
  if p.is_file():protected[str(p.relative_to(ROOT))]=sha(p)
 pool=[ROOT/'outputs/transformer_expanded_v2_20261005/data',ROOT/'outputs/sam2_g1_matched97_20261006/dataset',ROOT/'outputs/sam2_g1_factor_supplement_20261006/dataset']
 lookup={};fields=['xy','boxes','H','R','K','cam','seconds','raw','raw_valid']
 for folder in pool:
  prefix='' if folder.name=='data' else 'all_eligible_'
  dp=folder/f'{prefix}dataset.npz';mp=folder/f'{prefix}samples.json'
  data=dict(np.load(dp));meta=read(mp)
  for p in [dp,mp]:protected[str(p.relative_to(ROOT))]=sha(p)
  for i,m in enumerate(meta):
   assert m['key'] not in lookup
   lookup[m['key']]=(m,{k:data[k][i] for k in fields})
 selected=read(DESIGN/'analysis_weights.json');meta=[];arrays=[]
 for w in selected:
  m,d=lookup[w['key']];assert m['recording']==w['recording']
  assert len(m['frames'])==21 and len(d['xy'])==21
  assert np.allclose(d['xy'],(d['boxes'][:,:2]+d['boxes'][:,2:])/2,atol=1e-4)
  meta.append(dict(m,analysis_weight=w['weight'],levels=w['levels'],features=w['features']));arrays.append(d)
 data={k:np.stack([r[k] for r in arrays]) for k in fields}
 assert len(meta)==214 and len({m['key'] for m in meta})==214
 folds=[];cf={f['fold']:f for f in read(BANK/'camera/folds.json')}
 for f in read(BANK/'folds.json'):
  idx=[i for i,m in enumerate(meta) if m['recording']==f['test']]
  if not idx:continue
  c=cf[f['fold']];test=f['test']
  assert test==c['test'] and test not in f['train']+[f['validation']] and test not in c['train']+[c['validation']]
  folds.append(dict(fold=f['fold'],test=test,indices=idx,original_test_indices=f['test_indices']))
  for s in SEEDS:
   files=[BANK/f"fold_{f['fold']}/seed_{s}/{v}/checkpoint.pt" for v in VARIANTS]+[BANK/f"camera/fold_{f['fold']}/seed_{s}/full/camera.pt"]
   for p in files:protected[str(p.relative_to(ROOT))]=sha(p)
 for p in [BANK/'protocol.json',BANK/'folds.json',BANK/'camera/folds.json',BANK/'data/dataset.npz',BANK/'data/samples.json']:
  protected[str(p.relative_to(ROOT))]=sha(p)
 for p in (ROOT/'scripts/transformer_matched97_20261006').glob('*.py'):protected[str(p.relative_to(ROOT))]=sha(p)
 assert len(folds)==9 and sorted(i for f in folds for i in f['indices'])==list(range(214))
 np.savez_compressed(OUT/'dataset.npz',**data)
 write(OUT/'samples.json',meta);write(OUT/'folds.json',folds);write(OUT/'protected_sources.json',protected)
 protocol=read(DESIGN/'protocol.json')
 protocol.update(execution_version=OUT.name,user_authorization='Run the agreed design now',design_protocol_sha256=sha(DESIGN/'protocol.json'),bootstrap_replicates=2000,bootstrap_seed=20261006,bootstrap_interval='percentile 2.5/97.5%, exploratory, recording cluster pairs; fixed design weights, do not reoptimize',invalid_projection_penalty_pixels=32000,models=162,camera_models=27,seeds=SEEDS,variants=VARIANTS)
 write(OUT/'execution_protocol.json',protocol)
 write(OUT/'prepared.json',dict(windows=214,recordings=9,models=162,protected_files=len(protected),design_preserved=True,source_versions=dict(collections.Counter(m.get('source_version','unknown') for m in meta))))
 print(read(OUT/'prepared.json'),flush=True)
if __name__=='__main__':main()
