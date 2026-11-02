"""Freeze clear new labels and append them to the previous 199 rows exactly."""
from export import *

STATIC={('23','N10'),('23','N11'),('25','N03'),('25','N06'),('26','N04'),('26','N05'),('28','N05'),('28','N08'),('30','N05'),('30','N06'),('30','N07'),('31','N14')}

def main():
 dest=OUT/'dataset';items=read_json(dest/'clear_windows.json');data=[];meta=[];cache={}
 for m in items:
  cid=m['clip'];fs=np.array(m['frames']);cutoff=m['cutoff']
  if cid not in cache:
   cache[cid]=({(r['id'],r['frame']):r for r in v.load(OUT/cid/'final/masks_rle.jsonl.gz')},dict(np.load(OUT/cid/'input/reference.npz')),read_json(OUT/cid/'input/alignment.json'),{(r['id'],r['frame']):r for r in v.load(OUT/cid/'masks_rle.jsonl.gz')})
  lookup,cal,aa,rawrows=cache[cid];R,K,ns=cal['R'],cal['K'],cal['times_ns']
  # Temporal and reference-quality restrictions also apply to strict windows.
  if not all(aa[f]['status']=='valid' and aa[f]['quality']>=.5 for f in fs) or len({aa[f]['graph'] for f in fs})!=1:continue
  if not np.all((np.diff(ns[fs])/1e9>.08)&(np.diff(ns[fs])/1e9<.12)):continue
  boxes=np.array([lookup[m['id'],int(f)]['box'] for f in fs],float);points=fc.center(boxes);H=fc.homographies(R[fs],K);comp,z=fc.warp(points,H)
  if not np.isfinite(comp).all() or (z<=1e-6).any():continue
  raw=[];valid=[]
  for f in fs[:6]:
   r=rawrows.get((m['id'],int(f)));raw.append((np.array(r['box'][:2])+r['box'][2:])/2 if r else [0.,0.]);valid.append(float(r is not None))
  if not valid[-1]:raw[-1]=points[5];valid[-1]=1.
  data.append(dict(xy=points,boxes=boxes,H=H,R=R[fs],K=K,cam=fc.rotvec(fc.relative(R[fs])),seconds=(ns[fs]-ns[cutoff])/1e9,raw=raw,raw_valid=valid))
  source=read_json(OUT/cid/'input/metadata.json')['source']
  meta.append(dict(m,timestamp_ns=int(ns[cutoff]),nonoverlap=(cutoff-(m['start']+5))%25==0,source_version='g1_expansion_v2_20261005',source_start_s=source['start_s'],static_control=(cid,m['id']) in STATIC,edge=False))
 assert data
 new={k:np.stack([r[k] for r in data]).astype(np.float32) for k in data[0]}
 np.savez_compressed(dest/'clear_dataset.npz',**new);write_json(dest/'clear_samples.json',meta)
 base=ROOT/'outputs/transformer_expanded_20261005/data';old=dict(np.load(base/'dataset.npz'));oldmeta=read_json(base/'samples.json')
 assert len(oldmeta)==199 and old.keys()==new.keys()
 union={k:np.concatenate([old[k],new[k]]) for k in old};both=oldmeta+meta
 assert len({m['key'] for m in both})==len(both)
 checks={k:bool(np.array_equal(union[k][:199],old[k])) for k in old};assert all(checks.values())
 np.savez_compressed(dest/'combined_strict_dataset.npz',**union);write_json(dest/'combined_strict_samples.json',both)
 previous=[]
 for i in range(1,11):previous.append(read_json(ROOT/f'outputs/sam2_g1_v6_v8_20261001/{i:02d}/input/metadata.json')['source'])
 previous+=read_json(ROOT/'outputs/sam2_g1_expansion_20261005/selected_clips.json')
 selected=read_json(OUT/'selected_clips.json');overlap=[]
 for a in selected:
  for b in previous:
   if a['recording_group']==b['recording_group'] and max(a['start_s'],b['start_s'])<min(a['end_s'],b['end_s']):overlap.append([a['candidate'],b.get('candidate',b.get('clip'))])
 assert not overlap,overlap
 prior=read_json(OUT/'source_hashes.json');assert all(digest(ROOT/p)==h for p,h in prior.items())
 summary=read_json(dest/'summary.json');t=summary['totals'];oldrecs={m['recording'] for m in oldmeta};newrecs={m['recording'] for m in meta}
 t.update(clear_windows=len(meta),old_windows=199,combined_strict_windows=len(both),new_recordings=sorted(newrecs-oldrecs),union_recordings=len(oldrecs|newrecs),clear_window_static_controls=sum(m['static_control'] for m in meta),clear_sparse_windows=sum(m['nonoverlap'] for m in meta))
 # Remove inherited original-117 compatibility total; not used for this expansion.
 t.pop('union_compatible_windows',None)
 summary['protocol']['warning']='Offline reviewed automatic reference labels; overlapping windows are not independent cases. Static persons and G1 boundary clips remain explicit. Recording-held-out future camera models must be retrained for newly introduced exclusion pairs.'
 write_json(dest/'summary.json',summary)
 write_json(dest/'package_audit.json',dict(new_clear_windows=len(meta),old_windows=199,combined_strict_windows=len(both),old_arrays_exact=checks,old_metadata_exact=both[:199]==oldmeta,original_files_verified=len(prior),source_time_overlap=overlap,all_checks_passed=True))
 print(json.dumps(t,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
