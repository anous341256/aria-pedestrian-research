from bulk_common import *
import gc,traceback
sys.path.insert(0,str(ROOT/'scripts/depth_forecast_20261008'))
import extract as old

def main():
 assert (OUT/'protocol.json').exists();inv=read(OUT/'inventory.json');entries=sorted([e for e in inv['entries'] if e['targets']],key=lambda e:(e['recording'],e['rgb_timestamp_ns']));model=cfg=provider=mps=None;current=None;refs={};start=time.monotonic();failures=[];run_new=0
 for n,e in enumerate(entries):
  dest=OUT/'depth'/e['key'];dest.mkdir(parents=True,exist_ok=True)
  if (dest/'complete.json').exists():continue
  try:
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
    pair,timing=old.prepare_pair(provider,mps,e,ref);disp,lr=old.official.run_lr_consistency(model,pair['left'],pair['right'],cfg);raw=dest/'depth.npz';np.savez_compressed(raw,disparity_px=disp,lr_consistent=lr,**pair);run_new+=1
   ts=e['rgb_timestamp_ns'];assert max(timing['left_timestamp_ns'],timing['right_timestamp_ns'])<=ts and timing['left_timestamp_ns']<=ts-1000000 and ts-timing['left_timestamp_ns']<=40000000
   _,_,values=old.measure(e,pair,disp,lr);write(dest/'measurements.json',values);write(dest/'complete.json',dict(passed=True,timing=timing,raw_path=str(raw.relative_to(ROOT)),raw_sha256=sha(raw),cached=bool(e['cached_raw'])))
  except Exception as exc:
   # Keep candidate observations when a measurement is unavailable; never invent Z.
   failures.append(dict(key=e['key'],error=str(exc)));write(dest/'failure.json',dict(error=str(exc),traceback=traceback.format_exc()));values=[dict(target_key=t['target_key'],points=0,supported_points=0,lr_fraction=0.,z_median_m=None,z_iqr_m=None,usable=False,reasons=['extraction_failure']) for t in e['targets']];write(dest/'measurements.json',values);print('FAILED',e['key'],str(exc),flush=True)
  if (n+1)%10==0:
   progress=dict(processed=n+1,total=len(entries),new_inference_this_run=run_new,failures=len(failures),seconds=time.monotonic()-start,last=e['key']);write(OUT/'depth_progress.json',progress);print('DEPTH',json.dumps(progress),flush=True)
 vals=[v for e in entries for v in read(OUT/'depth'/e['key']/'measurements.json')];write(OUT/'depth_complete.json',dict(passed=True,frames=len(entries),targets=len(vals),usable=sum(v['usable'] for v in vals),seconds=time.monotonic()-start,failures=failures));print('DEPTH DONE',len(vals),flush=True)
if __name__=='__main__':main()
