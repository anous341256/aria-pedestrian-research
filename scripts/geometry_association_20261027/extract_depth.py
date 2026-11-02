import sys,time,gc
from assoc_common import *
sys.path.insert(0,str(ROOT/'scripts/depth_forecast_20261008'))
import extract as old
def main():
 plan=read(OUT/'inventory.json');model=cfg=provider=mps=None;current=None;refs={};start=time.monotonic()
 for n,e in enumerate(plan['entries']):
  dest=OUT/'depth'/e['key'];dest.mkdir(parents=True,exist_ok=True)
  if (dest/'complete.json').exists() or not e['targets']:continue
  if e['input'] not in refs:refs[e['input']]=dict(np.load(ROOT/e['input']/'reference.npz'))
  ref=refs[e['input']]
  if e['cached_raw']:
   raw=ROOT/e['cached_raw'];a=dict(np.load(raw));pair={k:a[k] for k in ['left','right','support_left','support_right','intrinsics','baseline_m','T_rgb_cv','T_world_cv','K_rgb']};disp=a['disparity_px'];lr=a['lr_consistent'];timing=read(ROOT/e['cached_complete'])['timing'];assert np.max(abs(pair['K_rgb']-ref['K']))<1e-8
  else:
   if model is None:
    import torch
    torch.set_num_threads(2);model,cfg=old.official.load_foundation_stereo(str(ROOT/'models/foundation_stereo_20261008/23-51-11/model_best_bp2.pth'))
   if current!=e['recording']:
    provider=mps=None;gc.collect();current=e['recording'];provider=old.official.data_provider.create_vrs_data_provider(str(ROOT/f'data/Aria/raw_vrs/{current}.vrs'));mps=old.official.MpsDataProvider(old.official.MpsDataPathsProvider(str(ROOT/f'data/Aria/raw_vrs/mps_{current}_vrs')).get_data_paths())
   pair,timing=old.prepare_pair(provider,mps,e,ref);disp,lr=old.official.run_lr_consistency(model,pair['left'],pair['right'],cfg);raw=dest/'depth.npz';np.savez_compressed(raw,disparity_px=disp,lr_consistent=lr,**pair)
  ts=e['rgb_timestamp_ns'];assert max(timing['left_timestamp_ns'],timing['right_timestamp_ns'])<=ts and timing['left_timestamp_ns']<=ts-1000000 and ts-timing['left_timestamp_ns']<=40000000
  _,_,values=old.measure(e,pair,disp,lr);write(dest/'measurements.json',values);write(dest/'complete.json',dict(timing=timing,raw_path=str(raw.relative_to(ROOT)),raw_sha256=sha(raw)))
  write(OUT/'depth_progress.json',dict(processed=n+1,total=len(plan['entries']),seconds=time.monotonic()-start))
  if (n+1)%20==0:print('DEPTH',n+1,len(plan['entries']),round(time.monotonic()-start,1),flush=True)
 vals=[v for e in plan['entries'] if e['targets'] for v in read(OUT/'depth'/e['key']/'measurements.json')]
 write(OUT/'depth_complete.json',dict(passed=True,frames=sum(bool(e['targets']) for e in plan['entries']),targets=len(vals),usable=sum(v['usable'] for v in vals),seconds=time.monotonic()-start))
 print(read(OUT/'depth_complete.json'),flush=True)
if __name__=='__main__':main()
