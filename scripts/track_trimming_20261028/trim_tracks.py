import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'partial_appearance_20261028'))
from partial_common import *
import cv2
cv2.setNumThreads(1)
def mask_quality(t):
 a=decode(t['rle']);x1,y1,x2,y2=map(int,t['box']);m=a[y1:y2,x1:x2];area=int(a.sum());n,l,stats,c=cv2.connectedComponentsWithStats(m,8);component=float(stats[1:,cv2.CC_STAT_AREA].max()/max(1,area)) if n>1 else 0.;return dict(area=area,occupancy=area/max(1,(x2-x1)*(y2-y1)),largest_component=component,horizontal_border=bool(x1<=25.6 or x2>=2534.4))
def frame_rule(r,q):
 reasons=[]
 if q['largest_component']<.65:reasons.append('fragmented_mask')
 if q['occupancy']<.15:reasons.append('sparse_mask')
 if r['action']=='matched':
  for flag,yes in [('weak_appearance',r['appearance']<.50),('high_cost',r['cost']>.65),('ambiguous_assignment',r['margin']<.02),('motion_conflict',r['mahalanobis2']>13.82),('depth_conflict',r['depth_standardized'] is not None and r['depth_standardized']>3)]:
   if yes:reasons.append(flag)
 ready=bool(not reasons and q['occupancy']>=.18 and q['largest_component']>=.75 and not q['horizontal_border'] and r['action']=='matched' and r['appearance']>=.55 and r['cost']<=.60 and r['margin']>=.03)
 return reasons,ready
def contiguous(rows):
 pieces=[];cur=[]
 for r in sorted(rows,key=lambda r:r['frame']):
  if cur and r['frame']!=cur[-1]['frame']+1:pieces.append(cur);cur=[]
  cur.append(r)
 if cur:pieces.append(cur)
 return pieces
def select(rows,qualities):
 groups=collections.defaultdict(list)
 for r in rows:groups[r['track_id']].append(r)
 result={'raw':[],'fixed_trim':[],'quality_trim':[]};audit=[]
 for tid,rr in groups.items():
  for run in contiguous(rr):
   result['raw'].append(run);result['fixed_trim'].append(run[2:-2] if len(run)>4 else []);pieces=[];cur=[];last=None
   for r in run:
    q=qualities[r['uid']];reasons,ready=frame_rule(r,q);ratio=None if last is None else q['area']/max(1,qualities[last['uid']]['area']);jump=ratio is not None and (ratio>2.5 or ratio<.4)
    if reasons or jump:
     if cur:pieces.append(cur);cur=[]
    if not reasons:cur.append(r)
    audit.append(dict(uid=r['uid'],frame=r['frame'],track_id=tid,hard_exclusion=reasons,break_before_area_ratio=bool(jump),area_ratio=ratio,endpoint_ready=ready,quality=q));last=r
   if cur:pieces.append(cur)
   for piece in pieces:
    cut=piece[2:-2] if len(piece)>4 else []
    while cut and not frame_rule(cut[0],qualities[cut[0]['uid']])[1]:cut=cut[1:]
    while cut and not frame_rule(cut[-1],qualities[cut[-1]['uid']])[1]:cut=cut[:-1]
    result['quality_trim'].append(cut)
 return result,audit
def package(cid,methods,ref):
 out={}
 for name,runs in methods.items():
  retained=[r for r in runs if len(r)>=21];segments=[];windows=[]
  for i,rr in enumerate(retained):
   segid=f'{cid}_{name}_{i+1:03d}';sources=[ref[r['uid']]['source_id'] for r in rr];segments.append(dict(segment_id=segid,clip=cid,original_track_id=rr[0]['track_id'],start=rr[0]['frame'],end=rr[-1]['frame'],length=len(rr),uids=[r['uid'] for r in rr],reference_ids_review_only=sources))
   for k in range(len(rr)-20):
    part=rr[k:k+21];windows.append(dict(segment_id=segid,start=part[0]['frame'],cutoff=part[15]['frame'],end=part[-1]['frame'],uids=[r['uid'] for r in part],reference_consistent_review_only=len({ref[r['uid']]['source_id'] for r in part})==1))
  links=sum(len(r)-1 for r in retained);sw=sum(sum(ref[a['uid']]['source_id']!=ref[b['uid']]['source_id'] for a,b in zip(r[:-1],r[1:])) for r in retained)
  out[name]=dict(segments=segments,windows=windows,summary=dict(segments=len(segments),retained_observations=sum(len(r) for r in retained),sliding21_windows=len(windows),stride21_windows=sum(len(r)//21 for r in retained),mixed_reference_windows=sum(not w['reference_consistent_review_only'] for w in windows),reference_transition_links=sw,reference_links=links,short_nonempty_pieces=sum(0<len(r)<21 for r in runs)),all_piece_lengths=[len(r) for r in runs])
 return out
def main():
 inv=read(BASE/'inventory.json');ref=read(BASE/'source_identity_reference.json');per={};start=time.monotonic();allwindows=[];allsegments=[]
 write(TRIM/'execution_code_hashes.json',{p.name:sha(p) for p in Path(__file__).parent.glob('*.py')})
 for cl in inv['clips']:
  cid=cl['clip'];rows=read(BASE/'tracking/natural'/ARM/f'{cid}.json')['rows'];targets={t['target_key']:t for e in inv['entries'] if e['clip']==cid for t in e['targets']};qualities={u:mask_quality(t) for u,t in targets.items()};methods,audit=select(rows,qualities);pack=package(cid,methods,ref);write(TRIM/'clips'/f'{cid}.json',dict(recording=cl['recording'],methods=pack,audit=audit));per[cid]=dict(recording=cl['recording'],**{m:r['summary'] for m,r in pack.items()});allsegments+=pack['quality_trim']['segments'];allwindows+=pack['quality_trim']['windows'];print('TRIM',cid,{m:r['summary'] for m,r in pack.items()},flush=True)
 totals={m:{k:sum(v[m][k] for v in per.values()) for k in next(iter(per.values()))[m]} for m in ['raw','fixed_trim','quality_trim']};write(TRIM/'comparison.json',dict(original_observations=inv['total_candidates'],total=totals,per_clip=per));write(TRIM/'retained_segments.json',allsegments);write(TRIM/'forecast_window_candidates.json',dict(history_points=16,future_points=5,rate_hz=10,not_merged_or_trained=True,windows=allwindows));write(TRIM/'run_complete.json',dict(passed=True,seconds=time.monotonic()-start));print('TOTAL',totals,flush=True)
if __name__=='__main__':main()
