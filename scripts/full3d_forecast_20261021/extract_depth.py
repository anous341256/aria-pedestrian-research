import sys,time,gc
from geometry_common import *
sys.path.insert(0,str(ROOT/'scripts/depth_forecast_20261008'))
import extract as old
def main():
 plan=read(OUT/'inventory.json');model=cfg=provider=mps=None;current=None;refs={};start=time.time()
 for n,e in enumerate(plan['entries']):
  dest=OUT/'frames'/e['key'];dest.mkdir(parents=True,exist_ok=True)
  if (dest/'complete.json').exists():continue
  ip=e['input']
  if ip not in refs:refs[ip]=dict(np.load(ROOT/ip/'reference.npz'))
  ref=refs[ip]
  if e['cached_raw']:
   raw=ROOT/e['cached_raw'];a=dict(np.load(raw));pair={k:a[k] for k in ['left','right','support_left','support_right','intrinsics','baseline_m','T_rgb_cv','T_world_cv','K_rgb']};disp=a['disparity_px'];lr=a['lr_consistent'];timing=read(ROOT/e['cached_complete'])['timing'];assert np.max(abs(pair['K_rgb']-ref['K']))<1e-8
  else:
   if model is None:
    import torch
    torch.set_num_threads(2);model,cfg=old.official.load_foundation_stereo(str(ROOT/'models/foundation_stereo_20261008/23-51-11/model_best_bp2.pth'))
   if current!=e['recording']:
    provider=mps=None;gc.collect();current=e['recording'];provider=old.official.data_provider.create_vrs_data_provider(str(ROOT/f'data/Aria/raw_vrs/{current}.vrs'));mps=old.official.MpsDataProvider(old.official.MpsDataPathsProvider(str(ROOT/f'data/Aria/raw_vrs/mps_{current}_vrs')).get_data_paths())
   pair,timing=old.prepare_pair(provider,mps,e,ref);disp,lr=old.official.run_lr_consistency(model,pair['left'],pair['right'],cfg);raw=dest/'depth.npz';np.savez_compressed(raw,disparity_px=disp,lr_consistent=lr,**pair)
  ts=e['rgb_timestamp_ns'];assert timing['rgb_timestamp_ns']==ts and max(timing['left_timestamp_ns'],timing['right_timestamp_ns'])<=ts and ts-timing['left_timestamp_ns']<=40000000
  z,valid,values=old.measure(e,pair,disp,lr);write(dest/'measurements.json',values);write(dest/'complete.json',dict(timing=timing,raw_path=str(raw.relative_to(ROOT)),raw_sha256=sha(raw)))
  write(OUT/'extraction_progress.json',dict(processed=n+1,total=len(plan['entries']),seconds=time.time()-start))
  if (n+1)%30==0:print('DEPTH',n+1,len(plan['entries']),round(time.time()-start,1),flush=True)
 cache={e['key']:{r['target_key']:r for r in read(OUT/'frames'/e['key']/'measurements.json')} for e in plan['entries']};z=np.zeros((117,21),np.float64);valid=np.zeros((117,21),bool);quality=np.zeros((117,21,2),np.float32)
 for s in plan['sample_refs']:
  for j,(key,tk) in enumerate(zip(s['frame_keys'],s['target_keys'])):
   r=cache[key][tk]
   if r['usable']:z[s['index'],j]=r['z_median_m'];valid[s['index'],j]=1;quality[s['index'],j]=[r['lr_fraction'],r['z_iqr_m']/r['z_median_m']]
 np.savez_compressed(OUT/'depth_labels.npz',z=z,valid=valid,quality=quality)
 write(OUT/'extraction_complete.json',dict(passed=True,frames=len(plan['entries']),history_valid=int(valid[:,:16].sum()),future_valid=int(valid[:,16:].sum()),future_total=117*5,history_windows_any=int(valid[:,:16].any(1).sum()),future_windows_any=int(valid[:,16:].any(1).sum())))
 print(read(OUT/'extraction_complete.json'),flush=True)
if __name__=='__main__':main()
