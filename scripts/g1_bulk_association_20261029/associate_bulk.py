from bulk_common import *
sys.path.insert(0,str(ROOT/'scripts/partial_appearance_20261028'))
from local_tracker import LocalTracker
from tracker import Tracker
from run_partial import measure,aggregate
sys.path.insert(0,str(ROOT/'scripts/track_trimming_20261028'))
from trim_tracks import mask_quality,frame_rule,contiguous

def load_clip(cl,entries):
 cid=cl['clip'];p=dict(np.load(OUT/'poses'/f'{cid}.npz'));a=np.load(OUT/'features'/f'{cid}.npz');emb=dict(zip(a['uids'],a['global_features']));feat={str(u):dict(features=a['features'][i,:n],xy=a['xy'][i,:n]) for i,(u,n) in enumerate(zip(a['uids'],a['counts']))};frames=[]
 for e in entries:
  dep={v['target_key']:v for v in read(OUT/'depth'/e['key']/'measurements.json')} if e['targets'] else {};obs=[]
  for t in e['targets']:
   u=t['target_key'];d=dep[u];b=np.array(t['box'],float);obs.append(dict(uid=u,box=b,uv=(b[:2]+b[2:])/2,embedding=emb[u].astype(float),z=d['z_median_m'] if d['usable'] else 0.,iqr=d['z_iqr_m'] if d['usable'] else 0.,depth_valid=d['usable']))
  frames.append(obs)
 return p,frames,feat

def infer(p,frames,features,partial=True):
 tr=LocalTracker(p['K'],p['R'][0],features) if partial else Tracker('rotation_translation_depth',p['K'],p['R'][0]);rows=[];logs=[]
 for f,obs in enumerate(frames):
  if partial:tr.audit_pairs=[]
  r,l=tr.step(obs,p['R'][f],p['c'][f],float((p['times_ns'][f]-p['times_ns'][0])/1e9),f);rows+=r;logs.append(l)
 return dict(rows=rows,frames=logs)

def package(cl,entries,rows,ref):
 targets={t['target_key']:t for e in entries for t in e['targets']};qualities={u:mask_quality(t) for u,t in targets.items()};groups=collections.defaultdict(list);audit={};windows=[]
 warnings=read(SOURCE/'manual_review_flags.json')['flags']
 for r in rows:groups[r['track_id']].append(r)
 for tid,rr in groups.items():
  for run in contiguous(rr):
   previous=None
   for r in run:
    q=qualities[r['uid']];reasons,ready=frame_rule(r,q);ratio=None if previous is None else q['area']/max(1,qualities[previous['uid']]['area']);jump=ratio is not None and (ratio>2.5 or ratio<.4);manual=[x['status'] for x in warnings if x['clip']==cl['clip'] and ('segment' not in x or x['segment']==ref[r['uid']]['source_segment'])];audit[r['uid']]=dict(frame=r['frame'],track_id=tid,quality=q,reasons=reasons,area_jump=bool(jump),manual_review_flags=manual,endpoint_ready=ready);previous=r
   for k in range(len(run)-20):
    part=run[k:k+21];uids=[r['uid'] for r in part];aa=[audit[u] for u in uids];reasons=sorted({x for a in aa for x in a['reasons']});manual=sorted({x for a in aa for x in a['manual_review_flags']});jump=any(a['area_jump'] for a in aa[1:]);windows.append(dict(key=f'{cl["clip"]}_T{tid}_{part[0]["frame"]:03d}',clip=cl['clip'],recording=cl['recording'],track_id=tid,frames=[r['frame'] for r in part],source_frames=[r['frame']+cl['lo'] for r in part],uids=uids,quality_reasons=reasons,area_jump=jump,manual_review_flags=manual,quality_eligible=not reasons and not jump and not manual,source_identity_consistent_review_only=len({ref[u]['source_id'] for u in uids})==1))
 return dict(windows=windows,frame_audit=audit)

def main():
 inv=read(OUT/'inventory.json');ref=read(OUT/'source_identity_reference.json');start=time.monotonic();by={c['clip']:[e for e in inv['entries'] if e['clip']==c['clip']] for c in inv['clips']};pending=list(inv['clips']);done=0
 while pending:
  advanced=False
  for cl in pending[:]:
   cid=cl['clip'];dest=OUT/'tracking'/f'{cid}.json'
   if dest.exists() and (OUT/'quality'/f'{cid}.json').exists():pending.remove(cl);done+=1;continue
   if not (OUT/'features'/f'{cid}.npz').exists() or not (OUT/'poses'/f'{cid}.npz').exists():continue
   entries=by[cid]
   if any(e['targets'] and not (OUT/'depth'/e['key']/'measurements.json').exists() for e in entries):continue
   p,frames,feat=load_clip(cl,entries);old=infer(p,frames,feat,False);new=infer(p,frames,feat,True);write(OUT/'tracking_baseline'/f'{cid}.json',old);write(dest,new);pack=package(cl,entries,new['rows'],ref);write(OUT/'quality'/f'{cid}.json',pack);pending.remove(cl);done+=1;advanced=True;write(OUT/'association_progress.json',dict(completed=done,total=len(inv['clips']),last=cid,seconds=time.monotonic()-start));print('ASSOC',cid,'candidates',len(new['rows']),'windows',len(pack['windows']),'quality',sum(w['quality_eligible'] for w in pack['windows']),flush=True)
  if not advanced and pending:time.sleep(10)
 byarm={'baseline':{},'partial':{}};windows=[]
 for cl in inv['clips']:
  cid=cl['clip'];byarm['baseline'][cid]=measure(read(OUT/'tracking_baseline'/f'{cid}.json')['rows'],ref,cl['recording']);byarm['partial'][cid]=measure(read(OUT/'tracking'/f'{cid}.json')['rows'],ref,cl['recording']);windows+=read(OUT/'quality'/f'{cid}.json')['windows']
 write(OUT/'comparison.json',{a:aggregate(v) for a,v in byarm.items()});write(OUT/'forecast_window_candidates.json',dict(history_points=16,future_points=5,rate_hz=10,not_merged_or_trained=True,windows=windows));write(OUT/'association_complete.json',dict(passed=True,clips=done,windows=len(windows),quality_eligible=sum(w['quality_eligible'] for w in windows),seconds_including_wait=time.monotonic()-start));print('ASSOCIATION DONE',len(windows),flush=True)
if __name__=='__main__':main()
