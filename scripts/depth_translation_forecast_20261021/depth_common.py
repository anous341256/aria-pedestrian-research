import json,hashlib,gzip
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/depth_translation_forecast_20261021'
BASE=ROOT/'outputs/translation_forecast_20261020'
OLDDEPTH=ROOT/'outputs/depth_forecast_20261008'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8');tmp.replace(p)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def inventory():
 meta=read(BASE/'samples.json');entries={};refs=[];protected={}
 for p in list(BASE.glob('*.json'))+[BASE/'dataset.npz']+list((BASE/'with_translation').rglob('checkpoint.pt'))+list((BASE/'with_translation').rglob('predictions.npz'))+list((ROOT/'scripts/translation_forecast_20261020').glob('*.py')):
  protected[str(p.relative_to(ROOT))]=sha(p)
 cache={}
 for i,m in enumerate(meta):
  cid=m['clip'];inp=ROOT/f'outputs/sam2_g1_v6_v8_20261001/{cid}/input';ann=ROOT/f'outputs/sam2_near_people_20261002/{cid}/final/masks_rle.jsonl.gz'
  if cid not in cache:
   with gzip.open(ann,'rt') as f:rows={(r['id'],r['frame']):r for r in map(json.loads,f)}
   ref=dict(np.load(inp/'reference.npz'));cache[cid]=(rows,ref)
   for p in [ann,inp/'reference.npz']:protected[str(p.relative_to(ROOT))]=sha(p)
  rows,ref=cache[cid];keys=[];targets=[]
  for j,f in enumerate(m['frames'][:16]):
   ts=int(ref['times_ns'][f]);assert ts==m['timestamps_ns'][j] and ts<=m['timestamp_ns']
   key=f"{m['recording']}_{ts}";tk=f"{cid}/{m['id']}/{f}";keys.append(key);targets.append(tk)
   if key not in entries:entries[key]=dict(key=key,recording=m['recording'],rgb_timestamp_ns=ts,historical=True,input=str(inp.relative_to(ROOT)),targets=[])
   e=entries[key]
   if not any(t['target_key']==tk for t in e['targets']):
    assert (m['id'],f) in rows
    e['targets'].append(dict(target_key=tk,rle=rows[(m['id'],f)]['rle']))
  refs.append(dict(index=i,key=m['key'],frame_keys=keys,target_keys=targets))
 entries=sorted(entries.values(),key=lambda x:(x['recording'],x['rgb_timestamp_ns']))
 for e in entries:
  p=OLDDEPTH/'frames'/e['key'];e['cache']=str(p.relative_to(ROOT)) if (p/'complete.json').exists() and (p/'depth.npz').exists() else None
 plan=dict(entries=entries,sample_refs=refs,windows=117,history_points=16,unique_frames=len(entries),cached=sum(e['cache'] is not None for e in entries),missing=sum(e['cache'] is None for e in entries))
 write(OUT/'inventory.json',plan);write(OUT/'source_hashes.json',protected)
 write(OUT/'protocol.json',dict(version=OUT.name,base=str(BASE.relative_to(ROOT)),windows=117,history_points=16,future_points=5,seeds=[17,29,43],folds=6,
  model='Reuse matching with_translation camera frozen; expand person GRU9->15, copy old weights, initialize six new input columns zero. Fine-tune all person weights in both paired arms.',
  arms=['no_depth','with_depth'],features=['log(Z)/log(30)','clip(delta log Z,-1,1)','valid','adjacent_pair_valid','LR_fraction','clip(IQR/Z,0,1)'],
  depth='Official sequential 512x512 vitl32 stereo; exposures<=historical RGB, left BEFORE RGB-1ms; gap<=40ms. Re-measure existing raw depth using exact current masks. >=30 points,LR>=.7,Z[.5,30],IQR/Z<=.25. Missing numbers zero; explicit validity. No window exclusion.',
  training='240 updates,batch48,AdamW lr.001 wd.01,clip1,original balanced draws and loss future+.5history; validation residual ADE every20,step0 eligible. No augmentation in either arm to keep mask depth associated with reviewed trajectory; adaptation matched, not direct replication of prior training.',
  comparison='Paired no/with depth isolates added input package. Saved88.24 model secondary reference. No camera retraining, future depth, future translation prediction or changed geometry.',
  source_labels='Offline SAM/MPS references; depth surface statistics are not independently measured human-center GT.'))
 print({k:v for k,v in plan.items() if k not in ['entries','sample_refs']},flush=True)
if __name__=='__main__':inventory()
