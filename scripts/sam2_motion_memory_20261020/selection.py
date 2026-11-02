"""Lock a recording-balanced, nonoverlapping existing near-person cohort."""
import json, hashlib, random, collections
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/sam2_motion_memory_20261020'
VERSION=OUT.name
SOURCES=['sam2_near_people_20261002','sam2_g1_expansion_20261005',
 'sam2_g1_expansion_v2_20261005','sam2_g1_matched97_20261006',
 'sam2_g1_factor_supplement_20261006','sam2_g1_uncovered200_20261006',
 'sam2_g1_remaining_all_20261007']
REJECTED={10,12,15,16,17,19,37,38,58,69,70,74,83}
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def write(p,x):
 p.parent.mkdir(parents=True,exist_ok=True)
 p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def load_sources():
 sources=[];forbidden=[]
 for name in SOURCES:
  d=ROOT/'outputs'/name
  listing=d/'selected_clips.json'
  if not listing.exists():
   listing=ROOT/'outputs/sam2_g1_v6_v8_20261001/selected_clips.json'
   if not listing.exists():continue
  for row in read(listing):
   cid=str(row.get('clip',str(row['candidate']).zfill(2)))
   cp=d/cid/'census.json'
   if not cp.exists():continue
   rec=row.get('recording_group','unknown_20260803_'+str(row['id']))
   start=int(row.get('source_start_frame',round(row['start_s']*10)))
   end=int(row.get('source_end_frame_exclusive',round(row['end_s']*10)))
   excluded=None
   rr=d/'rectified_selection_review.json'
   if rr.exists():excluded=read(rr).get('rejected',{}).get(cid)
   ir=d/'initial_review.json'
   if ir.exists():excluded=read(ir).get('exclude_clips',{}).get(cid)
   ledger=d/'clip_review_ledger.json'
   if ledger.exists():
    match=next((x for x in read(ledger) if x['clip']==cid),{})
    excluded=match.get('scene_exclusion') or excluded
   if name=='sam2_g1_uncovered200_20261006' and int(cid) in REJECTED:
    excluded='explicit_user_rejection'
   if excluded:
    forbidden.append(dict(recording=rec,start=start,end=end,reason=excluded,source=name,clip=cid))
    continue
   inp=d/cid/'input'
   if not (inp/'reference.npz').exists():inp=ROOT/'outputs/sam2_g1_v6_v8_20261001'/cid/'input'
   if not (inp/'reference.npz').exists():raise FileNotFoundError(inp)
   cal=np.load(inp/'reference.npz')
   assert len(cal['R'])==end-start,(name,cid,len(cal['R']),end-start)
   census=read(cp)
   if not census['segments']:continue
   sources.append(dict(source=name,clip=cid,recording=rec,start=start,end=end,
       input=str(inp.relative_to(ROOT)),census=str(cp.relative_to(ROOT)),
       people=len({s['id'] for s in census['segments']}),segments=census['segments']))
 return sources,forbidden
def main():
 assert not (OUT/'selection.json').exists(),'Selection already locked'
 sources,forbidden=load_sources();candidates=[];excluded=[]
 # Each old clip may yield multiple 2.1-4.0 second intervals. We disclose source
 # episode count separately; these intervals are not independent recordings.
 for src in sources:
  cal=np.load(ROOT/src['input']/'reference.npz')
  all_R=cal['R'];all_ts=cal['times_ns']/1e9
  good=np.ones(src['end']-src['start'],bool)
  for f in forbidden:
   if f['recording']==src['recording']:
    a=max(f['start'],src['start'])-src['start'];b=min(f['end'],src['end'])-src['start']
    if b>a:good[a:b]=False
  a=0
  while a<len(good):
   if not good[a]:a+=1;continue
   end=a
   while end<len(good) and good[end]:end+=1
   n=end-a
   if n>=21:
    for lo in range(a,end-20):
     hi=lo+21
     R=all_R[lo:hi];ts=all_ts[lo:hi]
     def angles(a,b):
      rel=np.swapaxes(a,-1,-2)@b
      return np.degrees(np.arccos(np.clip((np.trace(rel,axis1=-2,axis2=-1)-1)/2,-1,1)))
     stats=dict(max_1s_rotation_deg=float(angles(R[:-10],R[10:]).max()),
                omega_p95_deg_s=float(np.percentile(angles(R[:-1],R[1:])/np.diff(ts),95)))
     if stats['max_1s_rotation_deg']>20 or stats['omega_p95_deg_s']>50:continue
     entries=[]
     for s in src['segments']:
      rows=[x for x in s['rows'] if lo<=x['frame']<hi]
      if not rows:continue
      # Historical near-person rule: at least one >=12% observation, retention >=8%.
      seeds=[x for x in rows if x['box'][3]-x['box'][1]>=.12*1920]
      if not seeds:continue
      anchor=max(seeds,key=lambda x:(x.get('confidence',0),x['box'][3]-x['box'][1]))
      entries.append(dict(**{k:v for k,v in s.items() if k not in ['rows','start','end','anchor_frame','anchor_box']},
        rows=rows,start=max(int(lo),s['start']),end=min(int(hi)-1,s['end']),
        anchor_frame=anchor['frame'],anchor_box=anchor['box']))
     if entries:
      candidates.append(dict(**{k:v for k,v in src.items() if k!='segments'},lo=int(lo),hi=int(hi),
         absolute_start=src['start']+int(lo),absolute_end=src['start']+int(hi),entries=entries,camera_stats=stats))
   a=end
 # Reproducible ordering per recording; avoid performance-based selection.
 rng=random.Random(20261020);rng.shuffle(candidates)
 groups=collections.defaultdict(list)
 for c in candidates:groups[c['recording']].append(c)
 # Earliest-finish interval packing maximizes nonoverlapping coverage within
 # each recording. Random tie order resolves duplicate source representations.
 for rec in groups:groups[rec].sort(key=lambda c:c['absolute_end'],reverse=True)
 chosen=[];used=collections.defaultdict(list);discarded=[]
 while len(chosen)<200 and any(groups.values()):
  for rec in sorted(groups):
   while groups[rec]:
    c=groups[rec].pop()
    if any(c['absolute_start']<b and a<c['absolute_end'] for a,b in used[rec]):
     discarded.append(dict(source=c['source'],clip=c['clip'],lo=c['lo'],hi=c['hi'],reason='overlap_selected_interval'));continue
    c['id']=f'{len(chosen)+1:03d}';chosen.append(c);used[rec].append((c['absolute_start'],c['absolute_end']));break
   if len(chosen)>=200:break
 if len(chosen)<190:raise RuntimeError(f'Only {len(chosen)} nonoverlapping intervals; extend inventory before locking.')
 protected={}
 for c in chosen:
  for p in [ROOT/c['census'],ROOT/c['input']/'reference.npz',ROOT/c['input']/'rgb.mp4']:
   protected[str(p.relative_to(ROOT))]=digest(p)
 summary=dict(intervals=len(chosen),recordings=len(used),source_clips=len({(c['source'],c['clip']) for c in chosen}),
   frames=sum(c['hi']-c['lo'] for c in chosen),seconds=sum(c['hi']-c['lo'] for c in chosen)/10,
   segment_instances=sum(len(c['entries']) for c in chosen),
   source_counts=dict(collections.Counter(c['source'] for c in chosen)),
   recording_counts=dict(collections.Counter(c['recording'] for c in chosen)),
   candidate_intervals=len(candidates),overlap_discarded=len(discarded),
   note='Nonoverlapping intervals, not independent videos/people; existing reviewed proposals, no result-based selection.')
 write(OUT/'selection.json',chosen);write(OUT/'selection_summary.json',summary)
 write(OUT/'forbidden_intervals.json',forbidden);write(OUT/'protected_inputs.json',protected)
 write(OUT/'overlap_discarded.json',discarded)
 print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
