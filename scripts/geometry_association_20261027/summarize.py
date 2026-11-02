from assoc_common import *
import collections
def metrics(rows,ref):
 by_source=collections.defaultdict(list);by_track=collections.defaultdict(list)
 for r in rows:
  sid=ref[r['uid']]['source_id'];by_source[sid].append(r);by_track[r['track_id']].append(sid)
 changes=[];links=0
 for sid,rr in by_source.items():
  rr.sort(key=lambda r:r['frame'])
  for a,b in zip(rr[:-1],rr[1:]):
   links+=1
   if a['track_id']!=b['track_id']:changes.append(dict(source_id=sid,frame=b['frame'],previous_frame=a['frame'],previous_track=a['track_id'],current_track=b['track_id'],uid=b['uid'],action=b['action']))
 pairs=sum(len(v)*(len(v)-1)//2 for v in by_track.values());same=sum(sum(n*(n-1)//2 for n in collections.Counter(v).values()) for v in by_track.values())
 frag=sum(max(0,len({r['track_id'] for r in rr})-1) for rr in by_source.values());cont=sum(len(set(v))>1 for v in by_track.values());purity=sum(max(collections.Counter(v).values()) for v in by_track.values())
 return dict(candidate_assignments=len(rows),source_ids=len(by_source),output_tracks=len(by_track),source_link_opportunities=links,source_id_assignment_changes=len(changes),fragmentation_excess=frag,mixed_source_tracks=cont,source_majority_agreement=float(purity/max(1,len(rows))),same_track_pair_precision=float(same/max(1,pairs)),predicted_same_track_pairs=pairs,source_consistent_pairs=same,changes=changes,prior_types=dict(collections.Counter(r['prior_kind'] for r in rows)),depth_updates=dict(collections.Counter(r['depth_update'] for r in rows)),uncertain_matches=sum(r['action']=='matched' and not r['appearance_updated'] for r in rows))
def main():
 inv=read(OUT/'inventory.json');ref=read(OUT/'source_identity_reference.json');allresults={}
 for cond in ['natural','blackout3']:
  results={}
  for arm in ARMS:
   per={};rowsall=[]
   for cl in inv['clips']:
    rows=read(OUT/'tracking'/cond/arm/f'{cl["clip"]}.json')['rows'];per[cl['clip']]=dict(recording=cl['recording'],**metrics(rows,ref))
   additive=['candidate_assignments','source_ids','output_tracks','source_link_opportunities','source_id_assignment_changes','fragmentation_excess','mixed_source_tracks','predicted_same_track_pairs','source_consistent_pairs','uncertain_matches']
   total={k:sum(v[k] for v in per.values()) for k in additive};total['same_track_pair_precision']=total['source_consistent_pairs']/max(1,total['predicted_same_track_pairs']);total['source_majority_agreement']=sum(v['source_majority_agreement']*v['candidate_assignments'] for v in per.values())/max(1,total['candidate_assignments']);total['assignment_changes_per_100_links']=100*total['source_id_assignment_changes']/max(1,total['source_link_opportunities'])
   rec={}
   for g in sorted({v['recording'] for v in per.values()}):
    vv=[v for v in per.values() if v['recording']==g];sw=sum(v['source_id_assignment_changes'] for v in vv);nn=sum(v['source_link_opportunities'] for v in vv);rec[g]=dict(changes=sw,links=nn,changes_per_100_links=100*sw/max(1,nn))
   total['recording_equal_changes_per_100_links']=float(np.mean([r['changes_per_100_links'] for r in rec.values()]));total['prior_types']=dict(sum((collections.Counter(v['prior_types']) for v in per.values()),collections.Counter()));total['depth_updates']=dict(sum((collections.Counter(v['depth_updates']) for v in per.values()),collections.Counter()));results[arm]=dict(total=total,per_clip=per,per_recording=rec)
  allresults[cond]=results
 write(OUT/'comparison.json',allresults)
 for cond,results in allresults.items():print(cond,{a:v['total'] for a,v in results.items()},flush=True)
if __name__=='__main__':main()
