import json,hashlib,gzip
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/full3d_forecast_20261021'
BASE=ROOT/'outputs/translation_forecast_20261020'
DEPTH=ROOT/'outputs/depth_translation_forecast_20261021'
CAM=ROOT/'outputs/camera_full_data_20261002'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8');q.replace(p)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def lift(uv,z,K):
 ray=np.einsum('nij,ntj->nti',np.linalg.inv(K),np.concatenate([uv,np.ones(uv.shape[:-1]+(1,))],-1));return ray*z[...,None]
def to_fixed(x,A,b):return np.einsum('ntij,ntj->nti',A,x)+b
def to_camera(x,A,b):return np.einsum('ntji,ntj->nti',A,x-b)
def project(x,K):
 h=np.einsum('nij,ntj->nti',K,x);return h[...,:2]/np.where(abs(h[...,2:])>1e-12,h[...,2:],1e-12),x[...,2]
def inventory():
 meta=read(BASE/'samples.json');entries={};refs=[];cache={};protected={}
 for p in [BASE/'dataset.npz',BASE/'camera_dataset.npz',BASE/'samples.json',BASE/'folds.json',BASE/'camera_folds.json',CAM/'calibrations.json',CAM/'inventory.json']+list((BASE/'with_translation').rglob('checkpoint.pt'))+list((BASE/'with_translation').rglob('predictions.npz'))+list((ROOT/'scripts/translation_forecast_20261020').glob('*.py')):
  protected[str(p.relative_to(ROOT))]=sha(p)
 for i,m in enumerate(meta):
  cid=m['clip'];inp=ROOT/f'outputs/sam2_g1_v6_v8_20261001/{cid}/input';ann=ROOT/f'outputs/sam2_near_people_20261002/{cid}/final/masks_rle.jsonl.gz'
  if cid not in cache:
   with gzip.open(ann,'rt') as f:rows={(r['id'],r['frame']):r for r in map(json.loads,f)}
   cache[cid]=(rows,dict(np.load(inp/'reference.npz')))
   for p in [ann,inp/'reference.npz']:protected[str(p.relative_to(ROOT))]=sha(p)
  rows,ref=cache[cid];keys=[];tks=[]
  for j,f in enumerate(m['frames']):
   ts=int(ref['times_ns'][f]);assert ts==m['timestamps_ns'][j];key=f"{m['recording']}_{ts}";tk=f"{cid}/{m['id']}/{f}";keys.append(key);tks.append(tk)
   if key not in entries:entries[key]=dict(key=key,recording=m['recording'],rgb_timestamp_ns=ts,input=str(inp.relative_to(ROOT)),targets=[],historical=False)
   e=entries[key];e['historical']|=j<16
   if not any(t['target_key']==tk for t in e['targets']):e['targets'].append(dict(target_key=tk,rle=rows[(m['id'],f)]['rle']))
  refs.append(dict(index=i,frame_keys=keys,target_keys=tks))
 entries=sorted(entries.values(),key=lambda e:(e['recording'],e['rgb_timestamp_ns']))
 for e in entries:
  new=DEPTH/'frames'/e['key'];old=ROOT/'outputs/depth_forecast_20261008/frames'/e['key'];e['cached_raw']=None;e['cached_complete']=None
  if (new/'complete.json').exists():e['cached_raw']=read(new/'complete.json')['raw_path'];e['cached_complete']=str((new/'complete.json').relative_to(ROOT))
  elif (old/'complete.json').exists():e['cached_raw']=str((old/'depth.npz').relative_to(ROOT));e['cached_complete']=str((old/'complete.json').relative_to(ROOT))
 for p in [ROOT/'scripts/depth_forecast_20261008/extract.py',ROOT/'scripts/aria_stereo_validation_20261008/analyze.py',ROOT/'external/aria_stereo_20261008/stereo_utils.py']:protected[str(p.relative_to(ROOT))]=sha(p)
 plan=dict(entries=entries,sample_refs=refs,windows=117,unique_frames=len(entries),cached=sum(e['cached_raw'] is not None for e in entries),missing=sum(e['cached_raw'] is None for e in entries))
 write(OUT/'inventory.json',plan);write(OUT/'source_hashes.json',protected)
 write(OUT/'protocol.json',dict(version=OUT.name,windows=117,seeds=[17,29,43],folds=6,history=16,future=5,
  geometry='Xcam=Z K^-1 [u,v,1]; A=R0.T Rt; b=R0.T (ct-c0); Xfixed=A Xcam+b. Future Xcam=Ahat.T (Xhat_fixed-bhat); pixel=K Xcam divided by Z.',
  labels='SAM box-center ray times eroded mask surface median Z: approximate target proxy, not true body center. Future depth/MPS build labels ONLY. Missing depth retained and 3D supervision masked.',
  person='New GRU10->64/context68->95->64, future flow15/history flow48. History fixed XYZ relative latest valid history anchor, deltaXYZ,valid,time,LR,IQR/Z. Context anchorXYZ+anchor age. No raw camera input to person after geometry. 5 fixed draws,10 flow steps.',
  camera='Initialize matching old translation camera GRU6->32 + rotation head. Add future translation residual head32->64->15 initialized zero on top of historical OLS velocity. Fine-tune all camera parameters with equal normalized rotation and translation MSE using initial training-only error scales. Step0 eligible.',
  training='18 cameras on unchanged27185 full-camera windows,2000 updates batch256;18 new person fits240 updates batch48 recording-balanced. AdamW lr.001 wd.01 clip1. Person masked future-flow MSE+.5 masked history-flow MSE; validation valid3D ADE every20,step0 eligible. No image loss or data augmentation. Camera validation normalized joint MSE every100.',
  input_policy='Use only16historicalframes; last historical MPS <=forecast origin. Same prior up-to30ms start-boundary fallback for one full-camera history occurrence; pose remains before cutoff. Offline closed-loop references, not end-to-end online claim.',
  evaluation='All117 image ADE/FDE incl missing-depth-label windows; valid-only3D metrics and GT-person swaps. GT-camera swaps all. Joint/camera-rotation/translation swaps,hold,last3D,CV3D diagnostics; old88.24 secondary SYSTEM reference, not isolated depth ablation.',
  depth='Reuse exact raw stereo caches and remeasure masks. Official sequential512 vitl32, LR1px, historical exposures<=RGB, gap<=40ms. >=30pts,LR>=.7,Z.5..30,IQR/Z<=.25. No future fill.',
  preservation='No old models,labels,PPT overwritten; no selected baseline replacement or test tuning.'))
 print({k:v for k,v in plan.items() if k not in ['entries','sample_refs']},flush=True)
if __name__=='__main__':inventory()
