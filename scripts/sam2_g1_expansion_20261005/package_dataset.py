"""Package clear windows and compatible old+new union without training."""
from export import *

def main():
 dest=OUT/'dataset';clear=read_json(dest/'clear_windows.json');data=[];cache={};clear_meta=[]
 for m in clear:
  cid=m['clip'];fs=np.array(m['frames']);cutoff=m['cutoff']
  if cid not in cache:
   lookup={(r['id'],r['frame']):r for r in v.load(OUT/cid/'final/masks_rle.jsonl.gz')};cal=np.load(OUT/cid/'input/reference.npz');cache[cid]=(lookup,cal,read_json(OUT/cid/'input/alignment.json'))
  lookup,cal,aa=cache[cid];R,K,ns=cal['R'],cal['K'],cal['times_ns'];boxes=np.array([lookup[m['id'],int(f)]['box'] for f in fs],float);points=fc.center(boxes);H=fc.homographies(R[fs],K);comp,z=fc.warp(points,H)
  assert np.isfinite(comp).all() and (z>1e-6).all();assert all(aa[f]['status']=='valid' and aa[f]['quality']>=.5 for f in fs);assert len({aa[f]['graph'] for f in fs})==1
  assert np.all((np.diff(ns[fs])/1e9>.08)&(np.diff(ns[fs])/1e9<.12))
  data.append(dict(xy=points,boxes=boxes,H=H,R=R[fs],K=K,cam=fc.rotvec(fc.relative(R[fs])),seconds=(ns[fs]-ns[cutoff])/1e9))
  clear_meta.append(dict(m,timestamp_ns=int(ns[cutoff]),nonoverlap=(cutoff-(m['start']+5))%25==0))
 np.savez_compressed(dest/'clear_dataset.npz',**{k:np.stack([r[k] for r in data]).astype(np.float32) for k in data[0]});write_json(dest/'clear_samples.json',clear_meta)
 oldroot=ROOT/'outputs/biflow_aria_v1_20261002';old=np.load(oldroot/'dataset.npz');new=np.load(dest/'dataset.npz');keys=list(new.keys());union={k:np.concatenate([old[k],new[k]],axis=0) for k in keys}
 np.savez_compressed(dest/'combined_compatible_dataset.npz',**union)
 oldmeta=read_json(oldroot/'samples.json');newmeta=read_json(dest/'samples.json');unionmeta=[]
 for version,items in [('old_reviewed_20261002',oldmeta),('g1_expansion_20261005',newmeta)]:
  for m in items:unionmeta.append(dict(m,source_version=version,dataset_row=len(unionmeta)))
 assert len({m['key'] for m in unionmeta})==len(unionmeta)
 write_json(dest/'combined_compatible_samples.json',unionmeta)
 # Source-time overlap checked against every original near-person interval.
 selected=[r for r in read_json(OUT/'selected_clips.json') if f'{r["candidate"]:02d}' in CLIPS];previous=[]
 for i in range(1,11):previous.append(read_json(ROOT/f'outputs/sam2_g1_v6_v8_20261001/{i:02d}/input/metadata.json')['source'])
 overlaps=[]
 for a in selected:
  for b in previous:
   if a['recording_group']==b['recording_group'] and max(a['start_s'],b['start_s'])<min(a['end_s'],b['end_s']):overlaps.append([a,b])
 assert not overlaps
 allmedia=[]
 for path in sorted((OUT/'presentation').glob('*.mp4')):
  cap=cv2.VideoCapture(str(path));expected=700 if path.name.startswith('ALL') else int(next(r['duration_s'] for r in selected if path.name.startswith(f'{r["candidate"]:02d}_'))*10);count=0
  while True:
   ok,frame=cap.read()
   if not ok:break
   assert frame.shape==(1060,1280,3);count+=1
  cap.release();assert count==expected,(path,count,expected);allmedia.append(dict(path=path.name,decoded_frames=count,sha256=digest(path)))
 # Verify final donor points after every correction, including re-anchoring.
 points=0;overlap_alerts=0
 for cid in CLIPS:
  rows={(r['id'],r['frame']):r for r in v.load(OUT/cid/'final/masks_rle.jsonl.gz')};logs=v.load(OUT/cid/'final/decisions.jsonl.gz')
  for d in logs:
   if (d['id'],d['frame']) not in rows:continue
   for pt in d.get('negative_points',[]):
    donor=rows[pt['donor_id'],pt['frame']];x,y=map(int,pt['xy']);assert v.decode(donor['rle'])[y,x];assert pt['donor_id']!=d['id'];assert pt['frame']==d['frame'];points+=1
  ap=OUT/cid/('qa_delivery' if cid in ['13','15','18'] else 'qa_final')/'audit.json';overlap_alerts+=len(read_json(ap)['conflicts'])
 assert overlap_alerts==0
 original=read_json(OUT/'source_hashes.json');assert all(digest(ROOT/p)==h for p,h in original.items())
 audit=dict(clear_samples=len(clear),combined_compatible_samples=len(unionmeta),old_arrays_exact={k:bool(np.array_equal(union[k][:len(oldmeta)],old[k])) for k in keys},clear_nonoverlap_windows=sum(m['nonoverlap'] for m in clear_meta),original_files_verified=len(original),old_new_time_overlap=overlaps,final_valid_negative_points=points,final_large_overlap_alerts=overlap_alerts,media=allmedia,all_checks_passed=True)
 assert all(audit['old_arrays_exact'].values());write_json(dest/'package_audit.json',audit)
 print({k:v for k,v in audit.items() if k not in ['media','old_arrays_exact']},flush=True)
if __name__=='__main__':main()
