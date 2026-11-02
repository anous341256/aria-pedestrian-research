"""Sequential, ID-blind association. No source IDs or future observations accepted."""
import numpy as np
from scipy.optimize import linear_sum_assignment

UNMATCH=.72
def unit(x):return x/max(np.linalg.norm(x),1e-12)
def image_project(x,K):
 h=K@x
 if x[2]<=1e-6 or not np.isfinite(h).all():return None,None
 p=h[:2]/h[2];J=(K[:2]*h[2]-h[:2,None]*K[2])/h[2]**2
 return p,J
def warp_point(p,H):
 h=H@np.r_[p,1.]
 if h[2]<=1e-6:return None,None
 return h[:2]/h[2],(H[:2,:2]*h[2]-h[:2,None]*H[2,:2])/h[2]**2
def lift_world(uv,z,R,c,K):return R@(np.linalg.inv(K)@np.r_[uv,1.]*z)+c
def lift_covariance(uv,z,sz,su,R,K):
 ki=np.linalg.inv(K);ray=ki@np.r_[uv,1.];J=R@np.column_stack([z*ki[:,0],z*ki[:,1],ray])
 return J@np.diag([su**2,su**2,sz**2])@J.T+np.eye(3)*.02**2
def cv_predict(x,P,dt,acc):
 n=len(x)//2;F=np.eye(2*n);F[:n,n:]=np.eye(n)*dt;G=np.vstack([np.eye(n)*dt**2/2,np.eye(n)*dt]);return F@x,F@P@F.T+G@G.T*acc**2
def kalman_update(x,P,z,V):
 n=len(z);H=np.eye(n,len(x));S=H@P@H.T+V;gain=np.linalg.solve(S,H@P).T;xx=x+gain@(z-H@x);a=np.eye(len(x))-gain@H;PP=a@P@a.T+gain@V@gain.T;return xx,(PP+PP.T)/2
def pixel_sigma(box):return max(10.,.04*(box[3]-box[1]))
def size_cost(a,b):return float(min(1.,abs(np.log(max(1.,b[3]-b[1])/max(1.,a[3]-a[1])))/np.log(3)))

class Tracker:
 def __init__(self,arm,K,R0):
  self.arm=arm;self.K=K.copy();self.ki=np.linalg.inv(K);self.R0=R0.copy();self.tracks=[];self.next_id=1
 def H(self,R):return np.eye(3) if self.arm=='image_motion' else self.K@self.R0.T@R@self.ki
 def measurement2(self,o,R):
  p,J=warp_point(o['uv'],self.H(R));assert p is not None
  return p,J@J.T*pixel_sigma(o['box'])**2
 def measurement3(self,o,R,c):
  if not o['depth_valid']:return None,None
  z=o['z'];sz=max(.05,.05*z,o['iqr']/1.349)
  return lift_world(o['uv'],z,R,c,self.K),lift_covariance(o['uv'],z,sz,pixel_sigma(o['box']),R,self.K)
 def birth(self,o,R,c,time):
  p,V=self.measurement2(o,R);P=np.zeros((4,4));P[:2,:2]=V;P[2:,2:]=np.eye(2)*500**2
  t=dict(id=self.next_id,x2=np.r_[p,[0.,0.]],P2=P,x3=None,P3=None,last_depth=None,time=time,last_seen=time,hits=1,box=o['box'].copy(),anchor=o['embedding'].copy(),ema=o['embedding'].copy())
  if self.arm=='rotation_translation_depth':
   x,V3=self.measurement3(o,R,c)
   if x is not None:
    t['x3']=np.r_[x,[0.,0.,0.]];t['P3']=np.zeros((6,6));t['P3'][:3,:3]=V3;t['P3'][3:,3:]=np.eye(3)*3**2;t['last_depth']=time
  self.next_id+=1;self.tracks.append(t);return t
 def prior(self,t,R,c,time):
  dt=time-t['time'];assert dt>=0
  t['x2'],t['P2']=cv_predict(t['x2'],t['P2'],dt,200.);t['time']=time
  if t['x3'] is not None:t['x3'],t['P3']=cv_predict(t['x3'],t['P3'],dt,3.)
  uv,J=warp_point(t['x2'][:2],np.linalg.inv(self.H(R)))
  if uv is None:return dict(uv=None,kind='invalid_rotation_projection',cov=None,z=None,zvar=None)
  cov=J@t['P2'][:2,:2]@J.T;kind='image' if self.arm=='image_motion' else 'rotation';z=zvar=None;reason='no_3D_state'
  if self.arm=='rotation_translation_depth' and t['x3'] is not None:
   xc=R.T@(t['x3'][:3]-c);C=R.T@t['P3'][:3,:3]@R;uv3,j3=image_project(xc,self.K);age=time-t['last_depth']
   if age>.5:reason='stale_depth'
   elif uv3 is None:reason='behind_camera'
   elif np.sqrt(max(0.,C[2,2]))/xc[2]>.25:reason='uncertain_depth'
   else:
    cov3=j3@C@j3.T
    if np.sqrt(np.linalg.eigvalsh(cov3).max())>max(80.,.75*(t['box'][3]-t['box'][1])):reason='uncertain_projected_position'
    else:uv,cov,kind,z,zvar=uv3,cov3,'full3D',float(xc[2]),float(C[2,2]);reason='accepted'
  return dict(uv=uv,cov=cov,kind=kind,z=z,zvar=zvar,fallback_reason=reason)
 def pair(self,t,p,o,time):
  if p['uv'] is None:return 1e6,dict(reason='unprojectable')
  sim=float(np.clip(.3*np.dot(t['anchor'],o['embedding'])+.7*np.dot(t['ema'],o['embedding']),-1,1));delta=o['uv']-p['uv'];dist=float(np.linalg.norm(delta));S=p['cov']+np.eye(2)*pixel_sigma(o['box'])**2;d2=float(delta@np.linalg.solve(S,delta));gap=time-t['last_seen'];h=max(t['box'][3]-t['box'][1],o['box'][3]-o['box'][1]);reach=min(900.,max(120.,1.5*h)+150*gap)
  reason='ok'
  if sim<.45:reason='appearance_gate'
  elif dist>reach:reason='bounded_distance_gate'
  elif d2>13.82 and sim<.82:reason='motion_gate'
  mot=min(1.,np.sqrt(max(0.,d2)/13.82));dep=None
  if p['kind']=='full3D' and o['depth_valid']:
   sz=max(.05,.05*o['z'],o['iqr']/1.349);dep=abs(o['z']-p['z'])/np.sqrt(p['zvar']+sz**2);mot=.8*mot+.2*min(1.,dep/3)
  cost=.60*np.clip((1-sim)/.55,0,1)+.30*mot+.10*size_cost(t['box'],o['box'])
  return (float(cost) if reason=='ok' else 1e6),dict(reason=reason,appearance=sim,distance_px=dist,mahalanobis2=d2,depth_standardized=None if dep is None else float(dep),cost_unblocked=float(cost))
 def update(self,t,o,R,c,time,reliable):
  p,V=self.measurement2(o,R);t['x2'],t['P2']=kalman_update(t['x2'],t['P2'],p,V*(1 if reliable else 4));depth_status='not_full_arm'
  if self.arm=='rotation_translation_depth':
   x,V3=self.measurement3(o,R,c);depth_status='missing_depth'
   if x is not None and not reliable:depth_status='ambiguous_match_skip'
   elif x is not None:
    if t['x3'] is None:
     t['x3']=np.r_[x,[0.,0.,0.]];t['P3']=np.zeros((6,6));t['P3'][:3,:3]=V3;t['P3'][3:,3:]=np.eye(3)*9;t['last_depth']=time;depth_status='initialized'
    else:
     err=x-t['x3'][:3];d2=float(err@np.linalg.solve(t['P3'][:3,:3]+V3,err))
     if d2>16.27:depth_status='depth_innovation_rejected'
     else:
      t['x3'],t['P3']=kalman_update(t['x3'],t['P3'],x,V3);t['last_depth']=time;depth_status='updated'
  if reliable:t['ema']=unit(.9*t['ema']+.1*o['embedding'])
  t['last_seen']=time;t['hits']+=1;t['box']=o['box'].copy();return depth_status
 def step(self,observations,R,c,time,frame):
  # Whitelist copies ensure IDs, reference segments and arbitrary metadata cannot enter.
  keys=['uid','uv','box','embedding','z','iqr','depth_valid'];obs=[{k:o[k] for k in keys} for o in sorted(observations,key=lambda x:x['uid'])]
  self.tracks=[t for t in self.tracks if time-t['last_seen']<=1.0+1e-9]
  priors=[self.prior(t,R,c,time) for t in self.tracks];nt,nc=len(self.tracks),len(obs);cost=np.full((nt,nc+nt),UNMATCH);details={}
  for i,t in enumerate(self.tracks):
   for j,o in enumerate(obs):cost[i,j],details[i,j]=self.pair(t,priors[i],o,time)
  used=set();rows=[]
  if nt and nc:
   ri,cj=linear_sum_assignment(cost)
   for i,j in zip(ri,cj):
    if j>=nc or cost[i,j]>=UNMATCH:continue
    t=self.tracks[i];o=obs[j];p=priors[i];info=details[i,j];altrow=min([UNMATCH]+[float(cost[i,k]) for k in range(nc) if k!=j]);altcol=min([UNMATCH]+[float(cost[k,j]) for k in range(nt) if k!=i]);margin=min(altrow,altcol)-cost[i,j];reliable=bool(margin>=.08 and cost[i,j]<.50 and info['appearance']>=.70);gap=time-t['last_seen'];ds=self.update(t,o,R,c,time,reliable);used.add(j)
    rows.append(dict(frame=frame,uid=o['uid'],track_id=t['id'],action='matched',cost=float(cost[i,j]),margin=float(margin),appearance_updated=reliable,depth_update=ds,gap_s=float(gap),prior_kind=p['kind'],fallback_reason=p.get('fallback_reason'),predicted_uv=p['uv'].tolist(),prior_cov=p['cov'].tolist(),**info))
  for j,o in enumerate(obs):
   if j not in used:
    t=self.birth(o,R,c,time);rows.append(dict(frame=frame,uid=o['uid'],track_id=t['id'],action='birth',appearance_updated=False,prior_kind='birth',depth_update='initialized' if t['x3'] is not None else 'none'))
  rows.sort(key=lambda r:r['uid'])
  log=dict(frame=frame,time=time,prior_track_ids=[t['id'] for t in self.tracks[:nt]],candidate_uids=[o['uid'] for o in obs],costs=cost[:,:nc].tolist(),active_tracks=len(self.tracks),candidates=nc)
  for t in self.tracks:
   assert np.isfinite(t['x2']).all() and np.linalg.eigvalsh(t['P2']).min()>-1e-5
   if t['x3'] is not None:assert np.isfinite(t['x3']).all() and np.linalg.eigvalsh(t['P3']).min()>-1e-8
  assert len({r['track_id'] for r in rows})==len(rows)
  return rows,log
