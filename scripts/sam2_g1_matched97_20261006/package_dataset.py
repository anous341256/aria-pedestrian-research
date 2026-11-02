"""Select a 97-window new pool before training; retain all eligible surplus."""
from export import *
from window_motion import rotation_stats,eligible
STATIC={('35','N05'),('42','N02'),('42','N03'),('42','N04'),('42','N05'),('43','N06'),('47','N01'),('48','N06'),('51','N01')}

def spread(ids,n):
 return [ids[j] for j in np.linspace(0,len(ids)-1,n,dtype=int)] if n else []

def main():
 dest=OUT/'dataset';assert not (dest/'selection.json').exists()
 items=read_json(dest/'clear_windows.json');data=[];meta=[];cache={}
 for m in items:
  cid=m['clip'];fs=np.array(m['frames']);cutoff=m['cutoff']
  if cid not in cache:
   cache[cid]=({(r['id'],r['frame']):r for r in v.load(OUT/cid/'final/masks_rle.jsonl.gz')},dict(np.load(OUT/cid/'input/reference.npz')),read_json(OUT/cid/'input/alignment.json'),{(r['id'],r['frame']):r for r in v.load(OUT/cid/'masks_rle.jsonl.gz')})
  lookup,cal,aa,rawrows=cache[cid];R,K,ns=cal['R'],cal['K'],cal['times_ns']
  if not all(aa[f]['status']=='valid' and aa[f]['quality']>=.5 for f in fs) or len({aa[f]['graph'] for f in fs})!=1:continue
  if not np.all((np.diff(ns[fs])/1e9>.08)&(np.diff(ns[fs])/1e9<.12)):continue
  motion=rotation_stats(R[fs],ns[fs]/1e9)
  if not eligible(motion):continue
  boxes=np.array([lookup[m['id'],int(f)]['box'] for f in fs],float);points=fc.center(boxes);H=fc.homographies(R[fs],K);comp,z=fc.warp(points,H)
  if not np.isfinite(comp).all() or (z<=1e-6).any():continue
  raw=[];valid=[]
  for f in fs[:6]:
   r=rawrows.get((m['id'],int(f)));raw.append((np.array(r['box'][:2])+r['box'][2:])/2 if r else [0.,0.]);valid.append(float(r is not None))
  if not valid[-1]:raw[-1]=points[5];valid[-1]=1.
  data.append(dict(xy=points,boxes=boxes,H=H,R=R[fs],K=K,cam=fc.rotvec(fc.relative(R[fs])),seconds=(ns[fs]-ns[cutoff])/1e9,raw=raw,raw_valid=valid))
  source=read_json(OUT/cid/'input/metadata.json')['source']
  meta.append(dict(m,timestamp_ns=int(ns[cutoff]),nonoverlap=(cutoff-(m['start']+5))%25==0,source_version='g1_matched97_20261006',source_start_s=source['start_s'],static_control=(cid,m['id']) in STATIC,edge=False,motion_screen=motion))
 assert len(meta)>=97,len(meta)
 order=sorted(range(len(meta)),key=lambda i:(meta[i]['recording'],meta[i]['clip'],meta[i]['id'],meta[i]['cutoff']))
 static=[i for i in order if meta[i]['static_control']];moving=[i for i in order if not meta[i]['static_control']]
 nstatic=max(min(10,len(static)),97-len(moving));assert nstatic<=len(static) and len(moving)>=97-nstatic
 chosen=sorted(spread(static,nstatic)+spread(moving,97-nstatic))
 allarrays={k:np.stack([r[k] for r in data]).astype(np.float32) for k in data[0]}
 np.savez_compressed(dest/'all_eligible_dataset.npz',**allarrays);write_json(dest/'all_eligible_samples.json',meta)
 new={k:v[chosen] for k,v in allarrays.items()};selected=[meta[i] for i in chosen]
 np.savez_compressed(dest/'clear_dataset.npz',**new);write_json(dest/'clear_samples.json',selected)
 write_json(dest/'selection.json',dict(eligible=len(meta),chosen_indices=chosen,pool_size=97,static=nstatic,moving=97-nstatic,method='Sort recording/clip/identity/cutoff; uniform temporal spread separately in static and moving strata. No forecast scores inspected.',g1_rule='max 1-second rotation <=20 degrees AND p95 angular velocity <=50 deg/s; local operational heuristic, not a paper definition',source_quotas='Original recording quota could not be met with clear G1 windows. Recording and movement distributions remain different.'))
 base=ROOT/'outputs/transformer_expanded_20261005/data';old=dict(np.load(base/'dataset.npz'));oldmeta=read_json(base/'samples.json');assert len(oldmeta)==199 and old.keys()==new.keys()
 union={k:np.concatenate([old[k],new[k]]) for k in old};both=oldmeta+selected
 assert len({m['key'] for m in both})==296
 checks={k:bool(np.array_equal(union[k][:199],old[k])) for k in old};assert all(checks.values())
 np.savez_compressed(dest/'combined_strict_dataset.npz',**union);write_json(dest/'combined_strict_samples.json',both)
 previous=[read_json(ROOT/f'outputs/sam2_g1_v6_v8_20261001/{i:02d}/input/metadata.json')['source'] for i in range(1,11)]
 for ver in ['sam2_g1_expansion_20261005','sam2_g1_expansion_v2_20261005']:previous+=read_json(ROOT/f'outputs/{ver}/selected_clips.json')
 overlap=[]
 for a in read_json(OUT/'selected_clips.json'):
  for b in previous:
   if a['recording_group']==b['recording_group'] and max(a['start_s'],b['start_s'])<min(a['end_s'],b['end_s']):overlap.append([a['candidate'],b.get('candidate')])
 assert not overlap,overlap
 prior=read_json(OUT/'source_hashes.json');assert all(digest(ROOT/p)==h for p,h in prior.items())
 summary=read_json(dest/'summary.json');t=summary['totals'];t.update(all_eligible_windows=len(meta),clear_windows=97,old_windows=199,combined_strict_windows=296,clear_window_static_controls=nstatic,clear_sparse_windows=sum(m['nonoverlap'] for m in selected),contributing_clips=len({m['clip'] for m in selected}),contributing_recordings=len({m['recording'] for m in selected}))
 t.pop('union_compatible_windows',None)
 oldrecs={m['recording'] for m in oldmeta};newrecs={m['recording'] for m in selected}
 t.update(new_recordings=sorted(newrecs-oldrecs),union_recordings=len(oldrecs|newrecs))
 summary['protocol']['warning']='Offline reviewed automatic reference labels, not independent human GT. Overlapping windows are not independent cases. New 97 replaces old 97; fixed old199 evaluation, equal per-fold moving/static counts. Distribution differences remain.'
 write_json(dest/'summary.json',summary);write_json(dest/'package_audit.json',dict(all_checks_passed=True,old_arrays_exact=checks,old_metadata_exact=both[:199]==oldmeta,original_files_verified=len(prior),source_time_overlap=overlap,new_clear_windows=97))
 print(json.dumps(t,ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':main()
