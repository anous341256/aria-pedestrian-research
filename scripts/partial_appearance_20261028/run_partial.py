from local_tracker import *
from run import load_clip
from summarize import metrics
def infer(p,frames,features,strength=1.):
 tr=LocalTracker(p['K'],p['R'][0],features,strength);rows=[];logs=[];pairs=[]
 for f,obs in enumerate(frames):
  tr.audit_pairs=[];r,l=tr.step(obs,p['R'][f],p['c'][f],float((p['times_ns'][f]-p['times_ns'][0])/1e9),f);rows+=r;logs.append(l);pairs.extend(dict(frame=f,**a) for a in tr.audit_pairs)
 return dict(rows=rows,frames=logs,pairs=pairs)
def aggregate(by):
 keys=['candidate_assignments','source_ids','output_tracks','source_link_opportunities','source_id_assignment_changes','fragmentation_excess','mixed_source_tracks','within_track_reference_identity_changes','predicted_same_track_pairs','source_consistent_pairs','uncertain_matches'];total={k:sum(v[k] for v in by.values()) for k in keys};total['source_majority_agreement']=sum(v['source_majority_agreement']*v['candidate_assignments'] for v in by.values())/total['candidate_assignments'];total['same_track_pair_precision']=total['source_consistent_pairs']/max(1,total['predicted_same_track_pairs']);recs={}
 for rec in sorted({v['recording'] for v in by.values()}):
  vv=[v for v in by.values() if v['recording']==rec];n=sum(v['source_link_opportunities'] for v in vv);recs[rec]=dict(changes=sum(v['source_id_assignment_changes'] for v in vv),links=n,mixed_transitions=sum(v['within_track_reference_identity_changes'] for v in vv))
 total['recording_equal_changes_per_100_links']=float(np.mean([100*r['changes']/max(1,r['links']) for r in recs.values()]));return dict(total=total,per_clip=by,per_recording=recs)
def measure(rows,ref,recording):
 groups=collections.defaultdict(list)
 for r in rows:groups[r['track_id']].append(r)
 mixed=[dict(track_id=tid,frame=b['frame'],previous_frame=a['frame'],previous_source=ref[a['uid']]['source_id'],current_source=ref[b['uid']]['source_id']) for tid,rr in groups.items() for a,b in zip(rr[:-1],rr[1:]) if ref[a['uid']]['source_id']!=ref[b['uid']]['source_id']]
 return dict(recording=recording,**metrics(rows,ref),within_track_reference_identity_changes=len(mixed),mixed_events=mixed)
def main():
 inv=read(BASE/'inventory.json');ref=read(BASE/'source_identity_reference.json');by={'baseline':{},'partial':{}};start=time.monotonic();checks=[]
 write(OUT/'execution_code_hashes.json',{p.name:sha(p) for p in Path(__file__).parent.glob('*.py')})
 for cl in inv['clips']:
  cid=cl['clip'];p,frames=load_clip(cid);feat=load_features(cid);old=read(BASE/'tracking/natural'/ARM/f'{cid}.json');z=infer(p,frames,feat,0);assert z['rows']==old['rows'] and z['frames']==old['frames'];new=infer(p,frames,feat);write(OUT/'tracking'/f'{cid}.json',new);by['baseline'][cid]=measure(old['rows'],ref,cl['recording']);by['partial'][cid]=measure(new['rows'],ref,cl['recording']);checks.append(cid);print('TRACK',cid,'old/new changes',by['baseline'][cid]['source_id_assignment_changes'],by['partial'][cid]['source_id_assignment_changes'],flush=True)
 results={k:aggregate(v) for k,v in by.items()};write(OUT/'comparison.json',results);write(OUT/'run_complete.json',dict(passed=True,zero_strength_exact_clips=checks,seconds=time.monotonic()-start));print({a:r['total'] for a,r in results.items()},flush=True)
if __name__=='__main__':main()
