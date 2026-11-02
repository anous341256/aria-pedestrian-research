"""Lock the original separated dual-branch horizon experiment before scoring."""
from pathlib import Path
import sys,json,hashlib,gzip,copy
import numpy as np
import cv2
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/horizon_swap_20261017'
OLD=ROOT/'outputs/biflow_aria_v1_20261002'
BASE=ROOT/'outputs/factorized_forecast_v1_20261002'
CAM=ROOT/'outputs/camera_full_data_20261002'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.partial');tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');tmp.replace(p)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rm(x):return np.stack([cv2.Rodrigues(v)[0] for v in np.asarray(x,dtype=np.float64).reshape(-1,3)]).reshape(np.asarray(x).shape[:-1]+(3,3))
def rv(x):return np.stack([cv2.Rodrigues(v)[0][:,0] for v in np.asarray(x,dtype=np.float64).reshape(-1,3,3)]).reshape(np.asarray(x).shape[:-2]+(3,))
def warp(x,H):
 h=np.einsum('...ij,...j->...i',H,np.concatenate([x,np.ones(x.shape[:-1]+(1,))],-1));return h[...,:2]/h[...,2:],h[...,2]
def main():
 OUT.mkdir(parents=True,exist_ok=True)
 cfg=dict(version='horizon_swap_20261017',architecture='Original PPT separated camera GRU + pedestrian GRU/context/future-flow/history-flow. No cross-attention, gaze or depth.',
  arms={'long_history':{'history_points':16,'history_seconds':1.5,'future_points':5,'future_seconds':.5,'start':0},'short_history_control':{'history_points':6,'history_seconds':.5,'future_points':5,'future_seconds':.5,'start':10}},
  person_windows=117,seeds=[17,29,43],folds=6,camera_updates=2000,camera_batch=256,camera_validation_every=100,person_updates=240,person_batch=48,person_validation_every=20,learning_rate=.001,weight_decay=.01,gradient_clip=1.,history_loss_weight=.5,noise_sigma=.15,flow_steps=10,inference_samples=5,
  sampling='Camera repeated random permutations over original full-camera training window IDs; person recording-balanced original117 split. Identical paired indices across new arms.',
  split='Exact original six recording-held-out train/validation/test partitions. Camera full-data excludes both validation and test recordings.',
  window_rule='Reuse exact 21 chronological source frames. New cutoff is original frame index15, +1s after original cutoff. Long observes indices0..15; short observes10..15; both predict16..20. No reversal of time, no sample replacement.',
  normalization='Recompute camera relative rotations and homographies with new cutoff. Historical relative time divided by observation duration, following original convention. Rotation-only residual retains translation/parallax/noise.',
  training='Train both camera and person models from scratch. Camera validation rotation-vector MSE; pedestrian validation residual ADE. No final-image loss, no test tuning. Auxiliary history-head dimension follows history length, so parameter counts differ slightly.',
  augmentation='Original 40% reviewed,20% pre-review same-ID,40% Gaussian .005 image-size jitter with20% point masking; final observed point always present. Raw masks reloaded for full16 history.',
  comparison='Primary long vs short uses exactly same target frames and cutoff. Saved0.5->1.5 is descriptive only: target times,horizon,cutoff and output dimensions differ. Do not interpret their ADE reduction as architecture gain.',
  evaluation='Image ADE/FDE,residual ADE/FDE,camera SO3 angle,GT-camera/GT-person swaps,hold-last,raw CV,frozen-camera composition,recording-equal,per-seed,sparse and exploratory recording-bootstrap paired contrast.',
  labels='Offline SAM/MPS references, not independent per-frame GT or online system evaluation.')
 if (OUT/'protocol.json').exists():assert read(OUT/'protocol.json')==cfg;return
 meta=read(OLD/'samples.json');old=dict(np.load(OLD/'dataset.npz'));assert len(meta)==117
 protected=[OLD/'dataset.npz',OLD/'samples.json',BASE/'folds.json',BASE/'protocol.json',CAM/'dataset.npz',CAM/'folds.json',CAM/'protocol.json']
 folds=read(BASE/'folds.json');cfolds=read(CAM/'folds.json')
 rows=[];newmeta=[];cache={};rawcache={};rawchanged=0
 for i,m in enumerate(meta):
  cid=m['clip'];rpath=ROOT/f'outputs/sam2_g1_v6_v8_20261001/{cid}/input/reference.npz';rawpath=ROOT/f'outputs/sam2_near_people_20261002/{cid}/masks_rle.jsonl.gz'
  if cid not in cache:
   cache[cid]=dict(np.load(rpath));protected += [rpath,rawpath]
   with gzip.open(rawpath,'rt',encoding='utf-8') as fp:rawcache[cid]={(r['id'],r['frame']):r for r in map(json.loads,fp)}
  fs=np.asarray(m['frames']);ref=cache[cid];R=ref['R'][fs];K=ref['K'];ns=ref['times_ns'][fs]
  rel=np.einsum('ji,tjk->tik',R[15],R);H=np.einsum('ij,tjk,kl->til',K,rel,np.linalg.inv(K));xy=old['xy'][i].astype(np.float64)
  _,z=warp(xy,H);assert np.isfinite(z).all() and (z>1e-6).all()
  raw=xy.copy();valid=np.ones(21,dtype=np.float32)
  for j,f in enumerate(fs):
   r=rawcache[cid].get((m['id'],int(f)))
   if r is None:raw[j]=0;valid[j]=0
   else:box=np.asarray(r['box']);raw[j]=(box[:2]+box[2:])/2
  if not valid[15]:raw[15]=xy[15];valid[15]=1
  rawchanged+=int(((np.linalg.norm(raw[:16]-xy[:16],axis=1)>1)&valid[:16].astype(bool)).sum())
  rows.append(dict(xy=xy,raw=raw,raw_valid=valid,R=R,K=K,H=H,cam=rv(rel),seconds=(ns-ns[15])/1e9,boxes=old['boxes'][i]))
  mm=copy.deepcopy(m);mm.update(original_cutoff=m['cutoff'],cutoff=int(fs[15]),original_timestamp_ns=m['timestamp_ns'],timestamp_ns=int(ns[15]),timestamps_ns=ns.tolist());newmeta.append(mm)
 # Exact same full-camera windows; only their reference orientation changes.
 camera=dict(np.load(CAM/'dataset.npz'));r=rm(camera['rotation']);rel=np.einsum('nji,ntjk->ntik',r[:,15],r)
 camera['rotation']=rv(rel).astype(np.float32);camera['original_cutoff_ns']=camera['cutoff_ns'].copy()
 # Read actual sampled timestamps, rather than assuming exactly1 second.
 names=read(CAM/'inventory.json')['record_names'];cutoff=camera['cutoff_ns'].copy()
 for j,name in enumerate(names):
  pp=CAM/f'poses/{name}.npz';protected.append(pp);pose=np.load(pp);ii=np.flatnonzero(camera['record_index']==j);cutoff[ii]=pose['times_ns'][camera['frame_index'][ii]+10]
 camera['cutoff_ns']=cutoff;camera['frame_index']=camera['frame_index']+10
 for f,cf in zip(folds,cfolds):
  assert f['fold']==cf['fold'] and f['test']==cf['test'] and f['validation']==cf['validation']
  groups=[{meta[i]['recording'] for i in f[k]} for k in ['train_indices','val_indices','test_indices']]
  assert not(groups[0]&groups[1] or groups[0]&groups[2] or groups[1]&groups[2])
  for seed in cfg['seeds']:
   protected += [BASE/f'fold_{f["fold"]}/seed_{seed}/separated_dual/checkpoint.pt',BASE/f'fold_{f["fold"]}/seed_{seed}/separated_dual/predictions_reviewed.npz',CAM/f'fold_{f["fold"]}/seed_{seed}/full/camera.pt',CAM/f'fold_{f["fold"]}/seed_{seed}/full/predictions.npz']
 protected+=list((ROOT/'scripts/factorized_forecast_v1_20261002').glob('*.py'))+list((ROOT/'scripts/biflow_aria_v1_20261002').glob('*.py'))+list((ROOT/'scripts/camera_full_data_20261002').glob('*.py'))
 write(OUT/'source_hashes.json',{str(p.relative_to(ROOT)):sha(p) for p in set(protected)})
 np.savez_compressed(OUT/'dataset.npz',**{k:np.stack([r[k] for r in rows]) for k in rows[0]});np.savez_compressed(OUT/'camera_dataset.npz',**camera)
 write(OUT/'samples.json',newmeta);write(OUT/'folds.json',folds);write(OUT/'camera_folds.json',cfolds);write(OUT/'protocol.json',cfg)
 write(OUT/'preflight.json',dict(passed=True,person_windows=117,recordings=len(set(m['recording'] for m in meta)),full_camera_windows=len(camera['rotation']),raw_changed_historical_points=rawchanged,protected_files=len(set(protected)),source_interval_overlap_unchanged=True))
 print(read(OUT/'preflight.json'),flush=True)
if __name__=='__main__':main()
