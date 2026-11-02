"""Audit raw SAM outputs without overwriting them; prepare review-only candidates."""
from census import *
import gzip,collections,importlib.util,time,hashlib,os
VERSION=os.environ.get('SAM_REVIEW_VERSION','review_v1')
assert VERSION in ('review_v1','review_v2')
sp=importlib.util.spec_from_file_location('maskvis',ROOT/'scripts/sam2_incremental_v6_20261002/render.py');v=importlib.util.module_from_spec(sp);sp.loader.exec_module(v)

def dump(p,rows):
 p.parent.mkdir(parents=True,exist_ok=True)
 with gzip.open(p,'wt',encoding='utf8') as f:
  for r in rows:f.write(json.dumps(r,ensure_ascii=False)+'\n')

def process(cid):
 folder=OUT/cid;dest=folder/VERSION
 if (dest/'audit.json').exists():return
 raw=v.load(folder/'masks_rle.jsonl.gz');logs=v.load(folder/'decisions.jsonl.gz');census=read(folder/'census.json');N=read(folder/'input/metadata.json')['frames'];known={(e['id'],f) for e in census['segments'] for f in range(e['start'],e['end']+1)}
 ld={(r['id'],r['frame']):r for r in logs};byframe=collections.defaultdict(list);bad=collections.defaultdict(list);conflicts=[];seen=set();numeric=[]
 for r in raw:
  key=r['id'],r['frame'];assert key in known and key not in seen and 0<=r['frame']<N;seen.add(key)
  m=v.decode(r['rle']);assert m.any();y,x=np.where(m);assert int(m.sum())==r['quality']['pixels'];b=[int(x.min()),int(y.min()),int(x.max())+1,int(y.max())+1]
  # Stored bbox has exclusive upper bounds, matching describe_mask.
  assert all(np.isfinite(r['box']));q=r['quality'];d=ld.get(key,{})
  reasons=[]
  if q['guide_iou']<.45:reasons.append('weak_guide_agreement')
  if q['outside']>.10:reasons.append('mask_outside_prompt')
  if d.get('status')=='V6_UNCERTAIN':reasons.append('uncertain_memory_observation')
  if (b[3]-b[1])<.08*1920:reasons.append('below_retention_scale')
  if reasons:bad[key]+=reasons
  sm=cv2.resize(m,(640,480),interpolation=cv2.INTER_NEAREST).astype(bool);byframe[r['frame']].append((r,sm))
  numeric.append(dict(id=r['id'],frame=r['frame'],area=int(m.sum()),box=b,guide_iou=q['guide_iou'],outside=q['outside']))
 edits=read(OUT/'post_review.json') if (OUT/'post_review.json').exists() else {}
 for e in census['segments']:
  for pid in e['proposal_ids']:
   reason=edits.get('reject_proposals',{}).get(cid,{}).get(str(pid))
   if reason:
    for f in range(e['start'],e['end']+1):bad[e['id'],f].append('agent_visual_reject: '+reason)
 if VERSION=='review_v2':
  visual=read(OUT/'mask_visual_review.json')
  rejects=visual.get('quarantine_ids',{}).get(cid,{})
  for r in raw:
   if r['id'] in rejects:bad[r['id'],r['frame']].append('agent_mask_visual_quarantine: '+rejects[r['id']])
 prebad=set(bad)
 for f,rows in byframe.items():
  for i,(a,am) in enumerate(rows):
   for b,bm in rows[i+1:]:
    if (a['id'],f) in prebad or (b['id'],f) in prebad:continue
    inter=np.count_nonzero(am&bm);ratio=inter/max(1,min(am.sum(),bm.sum()));iou=inter/max(1,np.count_nonzero(am|bm))
    if ratio>.5:
     conflicts.append(dict(frame=f,ids=[a['id'],b['id']],intersection_over_smaller=float(ratio),iou=float(iou)))
     for r in [a,b]:bad[r['id'],f].append('unresolved_pair_mask_overlap')
 rows=[dict(r,annotation_status='provisional_user_review_pending') for r in raw if (r['id'],r['frame']) not in bad]
 pending=[dict(id=k[0],frame=k[1],reasons=list(dict.fromkeys(rs))) for k,rs in sorted(bad.items())]
 current={(r['id'],r['frame']) for r in rows};dependencies=[]
 for r in rows:
  key=r['id'],r['frame']
  for p in ld.get(key,{}).get('negative_points',[]):
   if (p['donor_id'],p['frame']) not in current:dependencies.append(dict(id=key[0],frame=key[1],donor_id=p['donor_id'],reason='Raw donor not in provisional review selection; memory not rerun'))
 dump(dest/'masks_rle.jsonl.gz',rows);write(dest/'pending.json',pending);write(dest/'census.json',census)
 write(dest/'audit.json',dict(clip=cid,frames=N,raw_masks=len(raw),review_masks=len(rows),raw_ids=len({r['id'] for r in raw}),review_ids=len({r['id'] for r in rows}),expected_person_frames=len(known),missing_person_frames=len(known-seen),pending_masks=len(pending),conflicts=conflicts,donor_dependencies=dependencies,status_counts=dict(collections.Counter(d['status'] for d in logs)),scene_exclusion=census['scene_exclusion'],note='Geometry/overlap screen, not proof of identity or whole-body correctness. Raw masks preserved. Donor-dependency flags retained, not silently promoted to training GT.'))
 write(dest/'mask_metrics.json',numeric)
 print(cid,'AUDIT',len(raw),'->',len(rows),'review masks',flush=True)

if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--wait',action='store_true');p.add_argument('--clips',nargs='+');args=p.parse_args()
 for c in read(OUT/'selected_clips.json'):
  cid=c['clip']
  if args.clips and cid not in args.clips:continue
  if args.wait:
   while not(OUT/cid/'complete.json').exists():time.sleep(2)
  if(OUT/cid/'complete.json').exists():process(cid)
