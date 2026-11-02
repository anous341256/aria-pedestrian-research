"""Freeze a probability sample of raw-recording G1 motion candidates."""
from pathlib import Path
import json, csv, hashlib
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/g1_population_survey_20261006'
SEED=2026100617
def read(p): return json.loads(Path(p).read_text(encoding='utf8'))
def write(p,x): Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
def angle(a,b): return np.degrees(np.arccos(np.clip((np.einsum('nij,nij->n',a,b)-1)/2,-1,1)))
def main():
 OUT.mkdir(parents=True,exist_ok=True)
 assert not (OUT/'sample_locked.json').exists(), 'Do not redraw a locked sample'
 inventory=read(ROOT/'outputs/dataset_threeway_20261001/motion_inventory.json'); frame=[]; exclusions=[]; strata=[]
 for rec in inventory:
  sid=rec['id']
  if not rec.get('mps_source') or not Path(rec['mps_source']).exists():
   exclusions.append(dict(id=sid,reason='missing_MPS',seconds=rec['prior_export_duration_s'])); continue
  mapping=list(csv.DictReader(open(rec['timestamp_map'],encoding='utf-8-sig')))
  times=np.array([int(x['vrs_device_time_ns']) for x in mapping],np.int64)
  df=pd.read_csv(rec['mps_source']);ts=np.rint(df.tracking_timestamp_us.to_numpy()*1000).astype(np.int64)
  ix=np.searchsorted(ts,times).clip(1,len(ts)-1);ix=np.where(abs(ts[ix]-times)<abs(ts[ix-1]-times),ix,ix-1)
  rr=Rotation.from_quat(df[['qx_world_device','qy_world_device','qz_world_device','qw_world_device']].to_numpy()[ix]).as_matrix()
  dt=np.diff(times)/1e9; net=angle(rr[:-10],rr[10:]); steps=angle(rr[:-1],rr[1:])
  valid=(abs(ts[ix]-times)<=30e6)&(df.quality_score.to_numpy()[ix]>=.5); graph=df.graph_uid.to_numpy()[ix]
  local=[]
  for start in range(0,len(times)-20,5):
   if not valid[start:start+21].all() or len(set(graph[start:start+21]))!=1: continue
   if not ((dt[start:start+20]>.08)&(dt[start:start+20]<.12)).all(): continue
   mx=float(max(net[start:start+11]));omega=float(np.percentile(steps[start:start+20]/dt[start:start+20],95))
   if mx>20 or omega>50: continue
   local.append(dict(key=f'{sid}_{start:05d}',id=sid,recording=rec['recording'],start_frame=start,
                     start_s=start/10,cutoff_s=(start+5)/10,end_s=(start+20)/10,max_1s_rotation=mx,
                     omega_p95=omega,video=rec['video'],timestamp_map=rec['timestamp_map'],mps=rec['mps_source'],
                     vrs=str(ROOT/f'data/Aria/raw_vrs/{rec["recording"]}.vrs'),scene=rec['scene']))
  if not local:
   exclusions.append(dict(id=sid,reason='no_eligible_camera_windows',seconds=rec['prior_export_duration_s']))
   continue
  # Quantiles of chronological candidate positions, never of EDGC/person presence.
  for q,indices in enumerate(np.array_split(np.arange(len(local)),min(4,len(local)))):
   if not len(indices):continue
   name=f'{sid}_Q{q+1}'
   for i in indices: local[i]['stratum']=name
   strata.append(dict(stratum=name,id=sid,N=len(indices),n=1,keys=[local[i]['key'] for i in indices]))
  frame+=local
  print(sid,len(local),flush=True)
 target=min(160,len(frame)); rng=np.random.default_rng(SEED)
 # Largest deficit allocation: one per stratum first, remainder proportional to capacity.
 remaining=target-len(strata); capacity=np.array([s['N']-1 for s in strata]);ideal=capacity/max(1,capacity.sum())*remaining
 assigned=np.floor(ideal).astype(int)
 for j in np.argsort(-(ideal-assigned),kind='stable')[:remaining-int(assigned.sum())]:assigned[j]+=1
 bykey={r['key']:r for r in frame}; selected=[]
 for s,extra in zip(strata,assigned):
  s['n']+=int(extra);s['pi']=s['n']/s['N'];s['weight']=s['N']/s['n']
  chosen=sorted(rng.choice(s['keys'],size=s['n'],replace=False).tolist());s['chosen']=chosen
  for key in chosen:selected.append(dict(bykey[key],inclusion_probability=s['pi'],survey_weight=s['weight']))
 selected.sort(key=lambda r:(r['id'],r['start_frame']))
 for j,r in enumerate(selected):r['sample_id']=f'S{j+1:03d}'
 assert len(selected)==target and len({r['key'] for r in selected})==target
 assert np.isclose(sum(r['survey_weight'] for r in selected),len(frame))
 protocol=dict(seed=SEED,target_samples=target,frame_count=len(frame),recordings=len({r['id'] for r in frame}),
               strata=len(strata),source_recordings=len(inventory),scope='Existing 2026-08-03 Aria collection, not general-population G1.',
               unit='2-second,21-frame raw-video window on a fixed 0.5-second grid; features from first six points only.',
               frame='All 29 recordings inspected via existing timestamps/MPS, including previously used times; no exclusion for training membership.',
               camera_rule='All21 poses within30ms quality>=.5 same graph; dt .08-.12s; max1s turn<=20deg and p95 angular speed<=50deg/s.',
               sampling='Recording x up-to-four chronological candidate-quantile strata, SRS without replacement; allocation fixed before RGB/person/features. All windows have positive known probability.',
               estimator='Primary person-window distribution: inverse-inclusion weighted ratio over all measured G1 targets, no per-frame person cap. Also report time-window-equal and near-person subgroup.',
               weighting='Each target inherits sampled time-window N_h/n_h; no EDGC high/low balancing or outcome-based weights. Frame count is overlapping window opportunities, NOT duration.',
               visual_G1='Target-relative: real pedestrian, no major truncation/occlusion/identity ambiguity, weak direct crossing, clear observation. Stairs/escalator/conveyed person excluded. Darkness alone not automatic rejection if visible.',
               detection='YOLO11l person conf>=.12 full1536 every frame; 4tile1280 at cutoff5. Enumerate all candidates; visual census allows logged missed-target corrections, no top-k. Broad measurement floor4% image height; near comparable subgroup min8% across21.',
               tracking='Rotation-assisted adjacent-frame Hungarian detector association; visual identity review; no fabricated missing boxes. Candidate if cutoff person and full21 detections; incomplete records retained as nonresponse/visibility outcomes.',
               measurement='SAM2.1 large image masks independently box-prompted at six historical frames, shared encoding; same mask-bbox-center definitions. Short survey extraction differs from prior V6 propagation; detector-box sensitivity and measurement failures reported.',
               no_replacement='No replacing random windows for no people, complex scenes, poor tracks, or large feature/error values; census failures remain logged. No training or forecast ADE evaluation.',
               thresholds=read(ROOT/'outputs/g1_factor_inventory_20261006/thresholds_locked.json')['thresholds'],
               uncertainty='Stratified survey/design approximation plus recording-resample sensitivity, accounting for targets sharing sampled windows. Correlated adjacent windows and one-session capture limit generalization.',
               exclusions=exclusions)
 write(OUT/'protocol_locked.json',protocol);write(OUT/'population_frame.json',frame);write(OUT/'strata.json',strata);write(OUT/'sample_locked.json',selected)
 paths=[OUT/x for x in ['protocol_locked.json','population_frame.json','strata.json','sample_locked.json']]
 write(OUT/'design_manifest.json',{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
 print('LOCKED',len(frame),len(selected),len(strata),flush=True)
if __name__=='__main__':main()
