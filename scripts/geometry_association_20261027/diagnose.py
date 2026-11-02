from assoc_common import *
from run import load_clip
from tracker import Tracker
class DiagnosticTracker(Tracker):
 def pair(self,t,p,o,time):
  cost,info=super().pair(t,p,o,time)
  if o['uid']==self.target:self.diagnostics.append(dict(track_id=t['id'],prior_kind=p['kind'],fallback_reason=p.get('fallback_reason'),predicted_uv=None if p['uv'] is None else p['uv'].tolist(),predicted_z=p.get('z'),current_z=float(o['z']) if o['depth_valid'] else None,**info))
  return cost,info
def main():
 ref=read(OUT/'source_identity_reference.json');out=[]
 for case in read(OUT/'review_cases.json'):
  cid=case['clip'];f=case['frame'];sid=case['source_id'];p,frames=load_clip(cid);uid=next(k for k,v in ref.items() if v['clip']==cid and v['frame']==f and v['source_id']==sid);result={}
  for arm in ARMS:
   tr=DiagnosticTracker(arm,p['K'],p['R'][0]);tr.target=uid;tr.diagnostics=[];prior_id=None
   for j in range(f+1):
    rr,_=tr.step(frames[j],p['R'][j],p['c'][j],float((p['times_ns'][j]-p['times_ns'][0])/1e9),j)
    if j<f:
     for r in rr:
      if ref[r['uid']]['source_id']==sid:prior_id=r['track_id']
   original=read(OUT/'tracking'/'natural'/arm/f'{cid}.json')['rows'];assert rr==[r for r in original if r['frame']==f]
   result[arm]=dict(previous_track=prior_id,previous_track_candidate_test=next((v for v in tr.diagnostics if v['track_id']==prior_id),None),all_candidates=tr.diagnostics)
  out.append(dict(clip=cid,source_id=sid,frame=f,arms=result))
 write(OUT/'case_diagnostics.json',out)
 for c in out:print(c['clip'],c['source_id'],c['frame'],{a:r['previous_track_candidate_test'] for a,r in c['arms'].items()},flush=True)
if __name__=='__main__':main()
