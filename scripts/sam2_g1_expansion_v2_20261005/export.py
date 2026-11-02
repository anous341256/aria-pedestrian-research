"""Annotation audit plus continuous trajectories/window inventory, no training."""
from qa import *
import importlib.util
spec=importlib.util.spec_from_file_location('forecast_common',ROOT/'scripts/biflow_aria_v1_20261002/common.py');fc=importlib.util.module_from_spec(spec);spec.loader.exec_module(fc)

def chunks(fs):
 out=[]
 for f in sorted(fs):
  if not out or f!=out[-1][-1]+1:out.append([])
  out[-1].append(f)
 return out

def main():
 dest=OUT/'dataset';dest.mkdir(exist_ok=True);assert not (dest/'dataset.npz').exists()
 selected={f'{r["candidate"]:02d}':r for r in read_json(OUT/'selected_clips.json')};meta=[];arrays=[];runs=[];frames_out=[];summary=[];sources={};errors=[]
 for cid in CLIPS:
  folder=OUT/cid/'final';info=selected[cid];rr=v.load(folder/'masks_rle.jsonl.gz');ds=v.load(folder/'decisions.jsonl.gz');lookup={(r['id'],r['frame']):r for r in rr};logs={(d['id'],d['frame']):d for d in ds};census=read_json(folder/'census.json');cal=np.load(OUT/cid/'input/reference.npz');R,K,ns=cal['R'],cal['K'],cal['times_ns'];alignment=read_json(OUT/cid/'input/alignment.json');N=len(R)
  assert len(lookup)==len(rr);assert len(logs)==len(ds)
  for r in rr:
   m=v.decode(r['rle']);yy,xx=np.where(m);assert len(xx)==r['quality']['pixels'];assert r['box']==[int(xx.min()),int(yy.min()),int(xx.max())+1,int(yy.max())+1];assert 0<=r['frame']<N
  clipruns={};clipwindows={};stats=dict(clip=cid,recording=info['recording_group'],episode=info['episode_group'],seconds=N/10,frames=N,people=census['people'],census_intervals=len(census['segments']),masks=len(rr),pending=sum('PENDING' in d['status'] for d in ds),uncertain=sum('UNCERTAIN' in d['status'] for d in ds),scene=info['visual_observation'],classification=info['status'])
  for e in census['segments']:
   eligible={'compatible':[],'clear':[]}
   for guide in e['rows']:
    f=guide['frame'];r=lookup.get((e['id'],f));d=logs.get((e['id'],f),{});reasons=[]
    accepted=bool(r and 'UNCERTAIN' not in d.get('status','') and 'PENDING' not in d.get('status',''))
    if not accepted:reasons.append('missing_or_uncertain')
    if e.get('review_only_reason'):reasons.append(e['review_only_reason'])
    if r:
     b=np.array(r['box']);q=r['quality']
     if b[0]<=5 or b[1]<=5 or b[2]>=2555 or b[3]>=1915:reasons.append('image_border_truncation')
     if b[3]-b[1]<.08*1920:reasons.append('below_retention_scale')
     if q['guide_iou']<.45 or q['outside']>.1:reasons.append('mask_guide_disagreement')
    for ex in e.get('forecast_excluded_intervals',[]):
     if ex['start']<=f<=ex['end']:reasons.append(ex['reason'])
    if accepted:eligible['compatible'].append(f)
    if not reasons:eligible['clear'].append(f)
    frames_out.append(dict(clip=cid,id=e['id'],frame=f,segment=e['segment'],accepted=accepted,clear=not reasons,reasons=reasons,box=None if r is None else r['box'],status=d.get('status','MISSING')))
   for tier,fs in eligible.items():
    for rn,run in enumerate(chunks(fs)):
     row=dict(clip=cid,id=e['id'],segment=e['segment'],run=rn,tier=tier,start=run[0],end=run[-1],points=len(run),span_s=(run[-1]-run[0])/10,recording=info['recording_group'],episode=info['episode_group'],classification=info['status'],description=e['description']);runs.append(row)
     if len(run)<21:continue
     clipruns[tier]=clipruns.get(tier,0)+1
     for cutoff in range(run[0]+5,run[-1]-15+1,5):
      fs=np.arange(cutoff-5,cutoff+16);aa=[alignment[int(f)] for f in fs]
      if not all(a['status']=='valid' and a.get('quality',0)>=.5 for a in aa) or len({a['graph'] for a in aa})!=1:continue
      if not np.all((np.diff(ns[fs])/1e9>.08)&(np.diff(ns[fs])/1e9<.12)):continue
      boxes=np.array([lookup[e['id'],int(f)]['box'] for f in fs],float);points=fc.center(boxes);H=fc.homographies(R[fs],K);comp,z=fc.warp(points,H)
      if not np.isfinite(comp).all() or (z<=1e-6).any():continue
      clipwindows[tier]=clipwindows.get(tier,0)+1
      # Independent inventories, array for compatible set plus explicit clear flag.
      if tier=='clear':continue
      meta.append(dict(key=f'{cid}_{e["id"]}_{cutoff:03d}',clip=cid,id=e['id'],segment=e['segment'],recording=info['recording_group'],episode=info['episode_group'],classification=info['status'],cutoff=cutoff,frames=fs.tolist(),timestamp_ns=int(ns[cutoff]),source_start_s=info['start_s'],description=e['description'],nonoverlap=(cutoff-(run[0]+5))%25==0,clear=all(int(f) in eligible['clear'] for f in fs)))
      arrays.append(dict(xy=points,boxes=boxes,H=H,R=R[fs],K=K,cam=fc.rotvec(fc.relative(R[fs])),seconds=(ns[fs]-ns[cutoff])/1e9))
  stats.update(continuous_runs_21=clipruns,windows=clipwindows);summary.append(stats)
  for name in ['masks_rle.jsonl.gz','decisions.jsonl.gz','census.json','finish_review.json']:sources[str((folder/name).relative_to(ROOT))]=digest(folder/name)
  sources[str((OUT/cid/'input/reference.npz').relative_to(ROOT))]=digest(OUT/cid/'input/reference.npz')
 np.savez_compressed(dest/'dataset.npz',**{k:np.stack([r[k] for r in arrays]).astype(np.float32) for k in arrays[0]})
 write_json(dest/'samples.json',meta);write_json(dest/'continuous_runs.json',runs);write_json(dest/'frame_quality.json',frames_out);write_json(dest/'sources.json',sources)
 # Clear-only stride starts may differ from compatible; store its own windows manifest.
 strict=[]
 for r in runs:
  if r['tier']!='clear' or r['points']<21:continue
  for cutoff in range(r['start']+5,r['end']-15+1,5):strict.append(dict(**r,cutoff=cutoff,frames=list(range(cutoff-5,cutoff+16)),key=f'{r["clip"]}_{r["id"]}_{cutoff:03d}'))
 write_json(dest/'clear_windows.json',strict)
 previous=read_json(ROOT/'outputs/biflow_aria_v1_20261002/samples.json');oldrecs={m['recording'] for m in previous};newrecs={s['recording'] for s in summary}
 total=dict(clips=len(summary),seconds=sum(s['seconds'] for s in summary),frames=sum(s['frames'] for s in summary),clip_local_ids=sum(s['people'] for s in summary),census_intervals=sum(s['census_intervals'] for s in summary),masks=sum(s['masks'] for s in summary),pending=sum(s['pending'] for s in summary),uncertain=sum(s['uncertain'] for s in summary),compatible_windows=len(meta),clear_windows=len(strict),compatible_runs_21=sum(s['continuous_runs_21'].get('compatible',0) for s in summary),clear_runs_21=sum(s['continuous_runs_21'].get('clear',0) for s in summary),recordings=len(newrecs),new_recordings=sorted(newrecs-oldrecs),union_recordings=len(oldrecs|newrecs),old_windows=len(previous),union_compatible_windows=len(previous)+len(meta),compatible_clear_flags=sum(m['clear'] for m in meta),clear_window_static_controls=sum(m['clip']=='19' for m in strict))
 write_json(dest/'summary.json',dict(totals=total,clips=summary,protocol=dict(past_points=6,future_points=15,step_seconds=.1,stride=5,compatible='Same missing/uncertain-status rule as original 117; includes partial/boundary masks. For comparability only, not recommended unfiltered training.',clear='Compatible plus reviewed occlusion/bicycle/stairs/head-only exclusions, no image-border truncation, height >=8%, mask-guide IoU >=.45/outside <=.1. No model-performance filtering. Clear windows restart stride at each clear run.',warning='No masks or centers interpolated. Offline bidirectional reference labels, not independent human GT. Overlapping windows are not independent cases. Static control separate. Need new recording-held-out camera training for future evaluation.',arrays='Historical inputs/true future targets stored together for preparation only. Future states never inference inputs. No model training executed.')))
 original=read_json(OUT/'source_hashes.json');mismatch=[p for p,h in original.items() if digest(ROOT/p)!=h];assert not mismatch,mismatch
 write_json(dest/'integrity.json',dict(original_files_verified=len(original),original_mismatches=mismatch,mask_bbox_rle_checks='passed',sample_count=len(meta),final_source_hashes=sources))
 print(json.dumps(total,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
