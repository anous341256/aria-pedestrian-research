"""Package every clear operational-G1 window; no quota, forecast scores or training."""
from export import *
from window_motion import rotation_stats,eligible
from reviewed_people import STATIC

def main():
 dest=OUT/'dataset';items=read_json(dest/'clear_windows.json');data=[];meta=[];cache={};rejected=[]
 for m in items:
  cid=m['clip'];fs=np.array(m['frames']);cutoff=m['cutoff']
  if cid not in cache:
   cache[cid]=({(r['id'],r['frame']):r for r in v.load(OUT/cid/'final/masks_rle.jsonl.gz')},dict(np.load(OUT/cid/'input/reference.npz')),read_json(OUT/cid/'input/alignment.json'),{(r['id'],r['frame']):r for r in v.load(OUT/cid/'masks_rle.jsonl.gz')})
  lookup,cal,aa,rawrows=cache[cid];R,K,ns=cal['R'],cal['K'],cal['times_ns']
  reasons=[]
  if not all(aa[f]['status']=='valid' and aa[f]['quality']>=.5 for f in fs) or len({aa[f]['graph'] for f in fs})!=1:reasons.append('pose_alignment')
  if not np.all((np.diff(ns[fs])/1e9>.08)&(np.diff(ns[fs])/1e9<.12)):reasons.append('time_step')
  motion=rotation_stats(R[fs],ns[fs]/1e9)
  if not eligible(motion):reasons.append('camera_G1_screen')
  boxes=np.array([lookup[m['id'],int(f)]['box'] for f in fs],float);points=fc.center(boxes);H=fc.homographies(R[fs],K);comp,z=fc.warp(points,H)
  if not np.isfinite(comp).all() or (z<=1e-6).any():reasons.append('invalid_projection')
  if reasons:rejected.append(dict(key=m['key'],reasons=reasons));continue
  raw=[];valid=[]
  for f in fs[:6]:
   r=rawrows.get((m['id'],int(f)));raw.append((np.array(r['box'][:2])+r['box'][2:])/2 if r else [0.,0.]);valid.append(float(r is not None))
  if not valid[-1]:raw[-1]=points[5];valid[-1]=1.
  data.append(dict(xy=points,boxes=boxes,H=H,R=R[fs],K=K,cam=fc.rotvec(fc.relative(R[fs])),seconds=(ns[fs]-ns[cutoff])/1e9,raw=raw,raw_valid=valid))
  source=read_json(OUT/cid/'input/metadata.json')['source']
  meta.append(dict(m,timestamp_ns=int(ns[cutoff]),source_version=OUT.name,source_start_s=source['start_s'],static_control=(cid,m['id']) in STATIC,edge=False,motion_screen=motion))
 assert data,'No eligible samples'
 assert len({m['key'] for m in meta})==len(meta)
 np.savez_compressed(dest/'all_eligible_dataset.npz',**{k:np.stack([r[k] for r in data]).astype(np.float32) for k in data[0]})
 write_json(dest/'all_eligible_samples.json',meta);write_json(dest/'strict_window_rejections.json',rejected)
 summary=read_json(dest/'summary.json');t=summary['totals']
 # The copied exporter also reports a union with the much older 117-window
 # trial. Remove those unrelated fields; the inventory computes the real union.
 for key in ['new_recordings','union_recordings','old_windows','union_compatible_windows','clear_window_static_controls']:
  t.pop(key,None)
 t.update(strict_g1_windows=len(meta),strict_g1_tracks=len({(m['clip'],m['id']) for m in meta}),strict_g1_clips=len({m['clip'] for m in meta}),strict_g1_recordings=len({m['recording'] for m in meta}),strict_static_windows=sum(m['static_control'] for m in meta))
 summary['protocol']['warning']='Offline agent-reviewed SAM2 reference labels, not independent human GT. All eligible windows retained; overlapping windows are not independent. Model checkpoints must exclude the corresponding recording from training and validation before future error analysis.'
 write_json(dest/'summary.json',summary)
 print(json.dumps(t,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
