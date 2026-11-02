"""Read-only replay and single-decision diagnostics; never change production outputs."""
from pathlib import Path
import sys,copy,collections
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/geometry_association_20261027'))
from assoc_common import read,write,sha,decode,ARMS
from run import load_clip
from tracker import Tracker,warp_point,image_project,size_cost,UNMATCH
import numpy as np
from scipy.optimize import linear_sum_assignment
BASE=ROOT/'outputs/geometry_association_20261027'
OUT=ROOT/'outputs/geometry_association_failures_20261027'
ARM='rotation_translation_depth'
class Inspect(Tracker):
 def prior(self,t,R,c,time):
  p=super().prior(t,R,c,time);u,J=warp_point(t['x2'][:2],np.linalg.inv(self.H(R)));v=dict(track_id=t['id'],last_seen=t['last_seen'],gap_s=float(time-t['last_seen']),last_depth=t['last_depth'],depth_age_s=None if t['last_depth'] is None else float(time-t['last_depth']),prior_kind=p['kind'],fallback_reason=p.get('fallback_reason'),uv=None if p['uv'] is None else p['uv'].tolist(),cov=None if p['cov'] is None else p['cov'].tolist(),z=p.get('z'),zvar=p.get('zvar'),rotation_uv=None if u is None else u.tolist(),rotation_cov=None if u is None else (J@t['P2'][:2,:2]@J.T).tolist(),box=t['box'].tolist())
  if t['x3'] is not None:
   xc=R.T@(t['x3'][:3]-c);C=R.T@t['P3'][:3,:3]@R;pp,jj=image_project(xc,self.K);v.update(raw_3D_z=float(xc[2]),raw_3D_z_std=float(np.sqrt(max(0,C[2,2]))),raw_3D_uv=None if pp is None else pp.tolist())
  self.priors[t['id']]=v;return p
 def pair(self,t,p,o,time):
  cost,d=super().pair(t,p,o,time);dd=dict(track_id=t['id'],uid=o['uid'],blocked_cost=float(cost),anchor_similarity=float(t['anchor']@o['embedding']),ema_similarity=float(t['ema']@o['embedding']),candidate_uv=o['uv'].tolist(),candidate_box=o['box'].tolist(),candidate_depth_valid=bool(o['depth_valid']),candidate_z=float(o['z']) if o['depth_valid'] else None,candidate_iqr=float(o['iqr']) if o['depth_valid'] else None,**d)
  if 'appearance' in d:
   app=.60*np.clip((1-d['appearance'])/.55,0,1);mot=min(1.,np.sqrt(max(0.,d['mahalanobis2'])/13.82));dep=d['depth_standardized'];motion=.30*(mot if dep is None else .8*mot+.2*min(1.,dep/3));size=.10*size_cost(t['box'],o['box']);dd.update(appearance_cost=float(app),motion_cost=float(motion),size_cost=float(size),depth_cost_effect=float(motion-.30*mot))
   # Same track state and appearance, only replace current projection by its 2D rotation fallback.
   v=self.priors[t['id']];alt=dict(p,uv=np.array(v['rotation_uv']),cov=np.array(v['rotation_cov']),kind='rotation',z=None,zvar=None) if v['rotation_uv'] is not None else dict(p,uv=None)
   ac,ad=super().pair(t,alt,o,time);dd['same_state_rotation_only']=dict(blocked_cost=float(ac),**ad)
  self.pairs.append(dd);return cost,d
 def update(self,t,o,R,c,time,reliable):
  rec=dict(track_id=t['id'],uid=o['uid'],reliable=bool(reliable))
  x,V=self.measurement3(o,R,c)
  if x is not None and t['x3'] is not None:
   err=x-t['x3'][:3];rec.update(depth_innovation_mahalanobis2=float(err@np.linalg.solve(t['P3'][:3,:3]+V,err)),world_innovation_m=float(np.linalg.norm(err)))
  result=super().update(t,o,R,c,time,reliable);rec['result']=result;self.updates.append(rec);return result
def counterfactual(trace,mode):
 ids=trace['frame_log']['prior_track_ids'];uids=trace['frame_log']['candidate_uids'];n,m=len(ids),len(uids);cost=np.full((n,m+n),UNMATCH);pairs={(r['track_id'],r['uid']):r for r in trace['pairs']}
 for i,tid in enumerate(ids):
  for j,uid in enumerate(uids):
   r=pairs[tid,uid];v=r['blocked_cost'];reason=r['reason']
   if mode=='remove_appearance_gate' and reason=='appearance_gate':
    # Removing this gate does not bypass distance/motion checks that follow it.
    h=max(trace['priors'][str(tid)]['box'][3]-trace['priors'][str(tid)]['box'][1],r['candidate_box'][3]-r['candidate_box'][1]);reach=min(900,max(120,1.5*h)+150*trace['priors'][str(tid)]['gap_s'])
    if r['distance_px']<=reach and (r['mahalanobis2']<=13.82 or r['appearance']>=.82):v=r['cost_unblocked']
   elif mode=='remove_motion_gate' and reason=='motion_gate':v=r['cost_unblocked']
   elif mode=='without_depth_cost' and reason=='ok':v=r['cost_unblocked']-r.get('depth_cost_effect',0)
   elif mode=='same_state_rotation_only':v=r.get('same_state_rotation_only',{}).get('blocked_cost',v)
   elif mode=='exclude_stale_over_0.3s' and trace['priors'][str(tid)]['gap_s']>.300001:v=1e6
   cost[i,j]=v
 if n and m:
  ii,jj=linear_sum_assignment(cost);return {uids[j]:int(ids[i]) for i,j in zip(ii,jj) if j<m and cost[i,j]<UNMATCH}
 return {}
def main():
 OUT.mkdir(parents=True,exist_ok=True);inv=read(BASE/'inventory.json');ref=read(BASE/'source_identity_reference.json');comp=read(BASE/'comparison.json')['natural'][ARM];mix=read(BASE/'identity_diagnostics.json')['natural'][ARM]['events'];catalog=[];checks=[]
 protected={str(p.relative_to(ROOT)):sha(p) for p in list((ROOT/'scripts/geometry_association_20261027').glob('*.py'))+list((BASE/'tracking'/'natural'/ARM).glob('*.json'))+[BASE/'comparison.json',BASE/'source_identity_reference.json',BASE/'protocol.json']}
 for cl in inv['clips']:
  cid=cl['clip'];p,frames=load_clip(cid);tr=Inspect(ARM,p['K'],p['R'][0]);saved=read(BASE/'tracking'/'natural'/ARM/f'{cid}.json');last_source={};last_source_track={};traces=[]
  for f,obs in enumerate(frames):
   tr.priors={};tr.pairs=[];tr.updates=[];before=dict(last_source);previous_for_person=dict(last_source_track)
   rows,log=tr.step(obs,p['R'][f],p['c'][f],float((p['times_ns'][f]-p['times_ns'][0])/1e9),f)
   assert rows==[r for r in saved['rows'] if r['frame']==f] and log==saved['frames'][f]
   trace=dict(frame=f,rows=rows,frame_log=log,priors={str(k):v for k,v in tr.priors.items()},pairs=tr.pairs,updates=tr.updates,previous_track_source={str(k):v for k,v in before.items()});traces.append(trace)
   events=[dict(type='reference_ID_change',**v) for v in comp['per_clip'][cid]['changes'] if v['frame']==f]+[dict(type='mixed_reference_track',**v) for v in mix if v['clip']==cid and v['frame']==f]
   for e in events:
    sid=e.get('source_id',e.get('current_source'));uid=next(o['uid'] for o in obs if ref[o['uid']]['source_id']==sid);assigned=next(r for r in rows if r['uid']==uid);pt=previous_for_person.get(sid);tid=assigned['track_id'];selected=next((r for r in tr.pairs if r['track_id']==tid and r['uid']==uid),None);old=next((r for r in tr.pairs if r['track_id']==pt and r['uid']==uid),None)
    cf={mode:counterfactual(trace,mode).get(uid) for mode in ['remove_appearance_gate','remove_motion_gate','without_depth_cost','same_state_rotation_only','exclude_stale_over_0.3s']}
    catalog.append(dict(clip=cid,frame=f,event=e,source_id=sid,uid=uid,assigned=assigned,previous_person_track=pt,previous_person_track_source=before.get(pt),previous_person_pair=old,selected_pair=selected,selected_track_previous_source=before.get(tid),counterfactual_selected_track=cf,available_candidates=[dict(uid=o['uid'],source_id=ref[o['uid']]['source_id']) for o in obs]))
   for r in rows:last_source[r['track_id']]=dict(source_id=ref[r['uid']]['source_id'],frame=f,uid=r['uid']);last_source_track[ref[r['uid']]['source_id']]=r['track_id']
  write(OUT/'traces'/f'{cid}.json',traces);checks.append(dict(clip=cid,frames=len(frames),exact=True));print('REPLAY',cid,flush=True)
 write(OUT/'failure_catalog.json',catalog);write(OUT/'protected_sources.json',protected);write(OUT/'replay_verification.json',dict(passed=True,clips=checks,frames=sum(c['frames'] for c in checks),events=len(catalog),counterfactuals='Single decision only, same historical state. No counterfactual rollout, no parameter tuning or source-result changes.'))
 for r in catalog:
  q=r['previous_person_pair'];s=r['selected_pair'];print(r['clip'],r['frame'],r['event']['type'],r['source_id'],'old',r['previous_person_track'],'->',r['assigned']['track_id'],r['assigned']['action'],'oldpair',None if q is None else {k:q.get(k) for k in ['reason','appearance','distance_px','mahalanobis2','cost_unblocked']},'selected',None if s is None else {k:s.get(k) for k in ['reason','appearance','distance_px','cost_unblocked']},'cf',r['counterfactual_selected_track'])
if __name__=='__main__':main()
