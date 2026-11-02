from partial_common import *
from tracker import Tracker,pixel_sigma,size_cost
import cv2
cv2.setNumThreads(1)
def load_features(cid):
 a=np.load(OUT/'patches'/f'{cid}.npz');return {str(u):dict(features=a['features'][i,:n],xy=a['xy'][i,:n]) for i,(u,n) in enumerate(zip(a['uids'],a['counts']))}

def local_match(a,b):
 na,nb=len(a['xy']),len(b['xy']);empty=dict(valid=False,similarity=0.,coverage=0.,weight=0.,inliers=0,pairs=[])
 if min(na,nb)<6:return empty
 dots=a['features']@b['features'].T;ab=dots.argmax(1);ba=dots.argmax(0);ii=np.array([i for i in range(na) if ba[ab[i]]==i and dots[i,ab[i]]>=.55],int);jj=ab[ii]
 if len(ii)<6:return dict(empty,mutual_pairs=len(ii))
 cv2.setRNGSeed(20261028)
 M,mask=cv2.estimateAffinePartial2D(a['xy'][ii],b['xy'][jj],method=cv2.RANSAC,ransacReprojThreshold=21/224,maxIters=200,confidence=.99,refineIters=10)
 if M is None:return empty
 keep=mask.ravel().astype(bool);n=int(keep.sum());scale=float(np.hypot(M[0,0],M[1,0]));angle=float(np.degrees(np.arctan2(M[1,0],M[0,0])));coverage=n/min(na,nb);span=min(float(np.ptp(a['xy'][ii[keep]],axis=0).max()),float(np.ptp(b['xy'][jj[keep]],axis=0).max())) if n else 0.;valid=bool(n>=6 and n/len(ii)>=.6 and coverage>=.15 and .4<=scale<=2.5 and abs(angle)<=25 and span>=.15);sim=float(np.median(dots[ii[keep],jj[keep]])) if n else 0.
 return dict(valid=valid,similarity=sim,coverage=coverage,weight=.5*min(1,coverage/.4) if valid else 0.,inliers=n,mutual_pairs=len(ii),scale=scale,angle=angle,pairs=np.column_stack([ii[keep],jj[keep]]).tolist(),transform=M.tolist())

class LocalTracker(Tracker):
 def __init__(self,K,R0,features,strength=1.):
  super().__init__(ARM,K,R0);self.features=features;self.strength=strength;self.cache={};self.audit_pairs=[]
 def birth(self,o,R,c,time):
  t=super().birth(o,R,c,time);t['gallery']=[(o['uid'],time)];return t
 def update(self,t,o,R,c,time,reliable):
  result=super().update(t,o,R,c,time,reliable)
  if reliable and time-t['gallery'][-1][1]>=.299999 and len(self.features[o['uid']]['xy'])>=6:
   t['gallery'].append((o['uid'],time));t['gallery']=[t['gallery'][0]]+t['gallery'][1:][-2:]
  return result
 def pair(self,t,p,o,time):
  cost,d=super().pair(t,p,o,time)
  if self.strength==0 or p['uv'] is None:return cost,d
  best=None;uid=o['uid']
  for previous,pt in t['gallery']:
   assert pt<time+1e-9
   key=(previous,uid)
   if key not in self.cache:self.cache[key]=local_match(self.features[previous],self.features[uid])
   r=self.cache[key]
   if r['valid'] and (best is None or r['similarity']>best['similarity']):best=dict(r,gallery_uid=previous,gallery_time=pt)
  original=d['appearance'];w=0. if best is None else self.strength*best['weight'];sim=original if best is None else float(np.clip((1-w)*original+w*best['similarity'],-1,1));h=max(t['box'][3]-t['box'][1],o['box'][3]-o['box'][1]);reach=min(900.,max(120.,1.5*h)+150*(time-t['last_seen']));reason='ok'
  if sim<.45:reason='appearance_gate'
  elif d['distance_px']>reach:reason='bounded_distance_gate'
  elif d['mahalanobis2']>13.82 and sim<.82:reason='motion_gate'
  # All non-appearance terms are retained exactly from the original implementation.
  newcost=d['cost_unblocked']-.60*np.clip((1-original)/.55,0,1)+.60*np.clip((1-sim)/.55,0,1)
  details=dict(d,appearance=sim,reason=reason,cost_unblocked=float(newcost),global_appearance=original,local_appearance=None if best is None else best['similarity'],local_weight=w,local_support=None if best is None else best)
  self.audit_pairs.append(dict(track_id=t['id'],uid=uid,gap_s=time-t['last_seen'],**details))
  return (float(newcost) if reason=='ok' else 1e6),details
