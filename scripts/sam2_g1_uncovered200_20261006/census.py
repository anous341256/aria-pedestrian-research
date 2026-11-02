"""Provisional near-person intervals; explicit review edits, no fabricated masks."""
from pathlib import Path
import json,math
import numpy as np,cv2
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/sam2_g1_uncovered200_20261006'
def read(p):return json.loads(Path(p).read_text('utf8'))
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2),'utf8')
def build(cid):
 folder=OUT/cid
 if (folder/'census.json').exists():return
 edits=read(OUT/'initial_review.json');props=read(folder/'refined/near_proposals.json');groups=[];used=set();rejected=edits.get('reject_proposals',{}).get(cid,{})
 if cid in edits.get('exclude_clips',{}):props=[]
 byid={t['proposal_id']:t for t in props}
 for alias in edits.get('merge_proposals',{}).get(cid,[]):
  groups.append([byid[i] for i in alias]);used.update(alias)
 groups.extend([[t] for t in props if t['proposal_id'] not in used and str(t['proposal_id']) not in rejected])
 N=read(folder/'input/metadata.json')['frames'];entries=[]
 for i,group in enumerate(groups,1):
  byframe={}
  for tr in group:
   limit=edits.get('proposal_ranges',{}).get(cid,{}).get(str(tr['proposal_id']),[0,N-1])
   for r in tr['rows']:
    if not limit[0]<=r['frame']<=limit[1]:continue
    score=r['confidence']*math.sqrt(r['box'][3]-r['box'][1])
    if r['frame'] not in byframe or score>byframe[r['frame']][0]:byframe[r['frame']]=(score,r)
  rows=[byframe[f][1] for f in sorted(byframe)];chunks=[]
  for r in rows:
   if not chunks or r['frame']-chunks[-1][-1]['frame']>4:chunks.append([])
   chunks[-1].append(r)
  for j,chunk in enumerate(chunks,1):
   lo,hi=chunk[0]['frame'],chunk[-1]['frame'];dense=[]
   for f in range(lo,hi+1):
    exact=next((r for r in chunk if r['frame']==f),None)
    if exact:dense.append(exact);continue
    left=max((r for r in chunk if r['frame']<f),key=lambda r:r['frame']);right=min((r for r in chunk if r['frame']>f),key=lambda r:r['frame']);t=(f-left['frame'])/(right['frame']-left['frame'])
    dense.append(dict(frame=f,box=((1-t)*np.array(left['box'])+t*np.array(right['box'])).tolist(),confidence=min(left['confidence'],right['confidence']),source='short_gap_prompt_only_interpolation',detection_id=None))
   anchor=max(chunk,key=lambda r:r['confidence']*math.sqrt(r['box'][3]-r['box'][1])*(.7 if min(r['box'][:2])<5 or r['box'][2]>2555 or r['box'][3]>1915 else 1))
   reviewed=all(f'{cid}:{t["proposal_id"]}' in edits.get('temporal_reviewed_proposals',[]) for t in group)
   entries.append(dict(id=f'N{i:02d}',segment=f'N{i:02d}_S{j:02d}',description='Provisional near-person identity',proposal_ids=[t['proposal_id'] for t in group],start=lo,end=hi,anchor_frame=anchor['frame'],anchor_box=anchor['box'],rows=dense,manual=None,
    review='Agent sampled temporal RGB review; user verification pending' if reviewed else 'Automatic rotation-assisted association; user identity verification pending',end_reason='clip_end' if hi==N-1 else 'proposal_visibility_end_pending_review'))
 entries.sort(key=lambda e:(-max(r['box'][3]-r['box'][1] for r in e['rows']),e['start']))
 for j,e in enumerate(entries):e['batch']=j//3+1
 write(folder/'census.json',dict(people=len({e['id'] for e in entries}),segments=entries,total_cap=None,batch_size=3,scene_exclusion=edits.get('exclude_clips',{}).get(cid),rejected_proposals=rejected,annotation_status='Provisional for user inspection; not training GT'))
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--clips',nargs='+');a=p.parse_args()
 for c in read(OUT/'selected_clips.json'):
  if (a.clips is None or c['clip'] in a.clips) and (OUT/c['clip']/'discovery_complete.json').exists():build(c['clip'])
