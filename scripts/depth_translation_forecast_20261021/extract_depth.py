import sys,time,gc,traceback
from depth_common import *
sys.path.insert(0,str(ROOT/'scripts/depth_forecast_20261008'))
import extract as old
def main():
 plan=read(OUT/'inventory.json');model=cfg=provider=mps=None;current=None;refs={};start=time.time();reused=0
 for n,e in enumerate(plan['entries']):
  dest=OUT/'frames'/e['key'];dest.mkdir(parents=True,exist_ok=True)
  if (dest/'complete.json').exists():continue
  ip=e['input']
  if ip not in refs:refs[ip]=dict(np.load(ROOT/ip/'reference.npz'))
  ref=refs[ip];begin=time.time()
  if e['cache']:
   p=ROOT/e['cache'];a=dict(np.load(p/'depth.npz'));timing=read(p/'complete.json')['timing'];pair={k:a[k] for k in ['left','right','support_left','support_right','intrinsics','baseline_m','T_rgb_cv','T_world_cv','K_rgb']};disp=a['disparity_px'];lr=a['lr_consistent'];reused+=1
   assert np.max(np.abs(pair['K_rgb']-ref['K']))<1e-8
   rawpath=str((p/'depth.npz').relative_to(ROOT))
  else:
   if model is None:
    import torch
    torch.set_num_threads(2)
    model,cfg=old.official.load_foundation_stereo(str(ROOT/'models/foundation_stereo_20261008/23-51-11/model_best_bp2.pth'))
   if current!=e['recording']:
    provider=mps=None;gc.collect();current=e['recording'];provider=old.official.data_provider.create_vrs_data_provider(str(ROOT/f'data/Aria/raw_vrs/{current}.vrs'));mps=old.official.MpsDataProvider(old.official.MpsDataPathsProvider(str(ROOT/f'data/Aria/raw_vrs/mps_{current}_vrs')).get_data_paths())
   pair,timing=old.prepare_pair(provider,mps,e,ref);disp,lr=old.official.run_lr_consistency(model,pair['left'],pair['right'],cfg)
   np.savez_compressed(dest/'depth.npz',disparity_px=disp,lr_consistent=lr,**pair);rawpath=str((dest/'depth.npz').relative_to(ROOT))
  ts=e['rgb_timestamp_ns'];assert max(timing['left_timestamp_ns'],timing['right_timestamp_ns'])<=ts and timing['left_timestamp_ns']<=ts-1000000
  assert abs(timing['left_timestamp_ns']-ts)<=40000000 and abs(timing['left_timestamp_ns']-timing['right_timestamp_ns'])<=1000000
  z,valid,values=old.measure(e,pair,disp,lr)
  write(dest/'measurements.json',values);write(dest/'complete.json',dict(timing=timing,raw_path=rawpath,raw_sha256=sha(ROOT/rawpath),cache=e['cache'],seconds=time.time()-begin))
  write(OUT/'extraction_progress.json',dict(processed=n+1,total=len(plan['entries']),reused_this_run=reused,seconds=time.time()-start))
  if (n+1)%10==0:print('PROGRESS',n+1,len(plan['entries']),round(time.time()-start,1),flush=True)
 build_features(plan)
def build_features(plan):
 cache={}
 for e in plan['entries']:cache[e['key']]={r['target_key']:r for r in read(OUT/'frames'/e['key']/'measurements.json')}
 features=np.zeros((117,16,6),np.float32);values=[]
 for s in plan['sample_refs']:
  prev=None
  for j,(key,tk) in enumerate(zip(s['frame_keys'],s['target_keys'])):
   r=cache[key][tk];values.append(dict(index=s['index'],frame=j,frame_key=key,**r));a=features[s['index'],j]
   if r['usable']:
    z=r['z_median_m'];a[:]=[np.log(z)/np.log(30),0,1,0,r['lr_fraction'],min(r['z_iqr_m']/z,1)]
    if prev is not None:a[1]=np.clip(np.log(z)-np.log(prev),-1,1);a[3]=1
    prev=z
   else:prev=None
 assert np.isfinite(features).all()
 np.savez_compressed(OUT/'depth_features.npz',features=features)
 write(OUT/'measurements.json',values)
 valid=features[:,:,2]>0
 write(OUT/'extraction_complete.json',dict(passed=True,unique_frames=len(plan['entries']),target_frame_occurrences=1872,valid=int(valid.sum()),valid_fraction=float(valid.mean()),windows_with_any=int(valid.any(1).sum()),windows_with_all=int(valid.all(1).sum()),valid_pair_count=int(features[:,:,3].sum())))
 print(read(OUT/'extraction_complete.json'),flush=True)
if __name__=='__main__':main()
