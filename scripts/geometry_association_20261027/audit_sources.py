"""Independent raw-disparity and raw-quaternion derivations."""
from assoc_common import *
import cv2,pandas as pd
def quaternion(q):
 q=q/np.linalg.norm(q,axis=1,keepdims=True);x,y,z,w=q.T
 return np.stack([1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w),2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w),2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)],-1).reshape(-1,3,3)
def main():
 inv=read(OUT/'inventory.json');maxR=maxc=maxstat=0.;count=0
 for rec in sorted({c['recording'] for c in inv['clips']}):
  df=pd.read_csv(ROOT/f'data/Aria/raw_vrs/mps_{rec}_vrs/slam/closed_loop_trajectory.csv',usecols=['tracking_timestamp_us']+[f't{a}_world_device' for a in 'xyz']+[f'q{a}_world_device' for a in 'xyzw']);ns=df.tracking_timestamp_us.to_numpy(np.int64)*1000
  for cl in inv['clips']:
   if cl['recording']!=rec:continue
   cid=cl['clip'];p=dict(np.load(OUT/'poses'/f'{cid}.npz'));cal=np.load(ROOT/f'outputs/sam2_g1_v6_v8_20261001/{cid}/input/reference.npz');ext=cal['T_device_camera'];j=np.searchsorted(ns,p['times_ns'],side='right')-1;rd=quaternion(df[[f'q{a}_world_device' for a in 'xyzw']].to_numpy()[j]);pos=df[[f't{a}_world_device' for a in 'xyz']].to_numpy()[j]+rd@ext[:3,3];pos-=pos[0].copy();maxR=max(maxR,float(abs(rd@ext[:3,:3]-p['R']).max()));maxc=max(maxc,float(abs(pos-p['c']).max()));assert np.array_equal(ns[j],p['selected_mps_ns'])
 for k,e in enumerate(inv['entries']):
  if not e['targets']:continue
  dest=OUT/'depth'/e['key'];done=read(dest/'complete.json');raw=ROOT/done['raw_path'];assert sha(raw)==done['raw_sha256'];a=dict(np.load(raw));disp=a['disparity_px'];y,x=np.indices(disp.shape);xr=x-disp;xi=np.clip(np.nan_to_num(xr),0,511).astype(int);support=a['support_left']&a['support_right'][y,xi]&(xr>=0)&(xr<=511);z=(float(a['baseline_m'])*float(a['intrinsics'][0])/(disp+1e-6)).astype(np.float32);valid=support&a['lr_consistent']&np.isfinite(z)&(z>0)&np.isfinite(disp)&(disp>0)
  yy,xx=np.nonzero(support&np.isfinite(z)&(z>0));zz=z[yy,xx];fx,fy,cx,cy=a['intrinsics'];p=np.column_stack(((xx-cx)*zz/fx,(yy-cy)*zz/fy,zz,np.ones(len(zz))))@a['T_rgb_cv'].T;h=p[:,:3]@a['K_rgb'].T;uv=h[:,:2]/h[:,2,None];keep=(p[:,2]>0)&np.isfinite(uv).all(1)&(uv[:,0]>=0)&(uv[:,0]<2559.5)&(uv[:,1]>=0)&(uv[:,1]<1919.5);xx=xx[keep];yy=yy[keep];p=p[keep];uv=np.rint(uv[keep]).astype(int);order=np.argsort(p[:,2],kind='stable');_,ii=np.unique(uv[order,1]*2560+uv[order,0],return_index=True);ii=order[ii];p=p[ii];uv=uv[ii];vv=valid[yy[ii],xx[ii]];actual={r['target_key']:r for r in read(dest/'measurements.json')}
  for t in e['targets']:
   r=t['rle'];x0,y0,x1,y1=map(int,r[-4:]);mask=np.zeros((1920,2560),np.uint8);mask[y0:y1+1,x0:x1+1]=np.repeat(np.arange(len(r[:-4]))%2,np.asarray(r[:-4],int)).reshape(y1-y0+1,x1-x0+1);mask=cv2.erode(mask,np.ones((15,15),np.uint8));member=mask[uv[:,1],uv[:,0]].astype(bool);vals=p[member&vv,2];n=len(vals);r=actual[t['target_key']];assert n==r['points'] and int(member.sum())==r['supported_points'];q=n/max(1,int(member.sum()));assert q==r['lr_fraction'];usable=False
   if n>=30:
    med=float(np.median(vals));iqr=float(np.diff(np.quantile(vals,[.25,.75]))[0]);maxstat=max(maxstat,abs(med-r['z_median_m']),abs(iqr-r['z_iqr_m']));usable=.5<=med<=30 and q>=.7 and iqr/med<=.25
   assert usable==r['usable'];count+=1
  timing=done['timing'];assert timing['rgb_timestamp_ns']==e['rgb_timestamp_ns'];assert max(timing['left_timestamp_ns'],timing['right_timestamp_ns'])<=e['rgb_timestamp_ns'];assert abs(timing['left_timestamp_ns']-timing['right_timestamp_ns'])<=1000000
  if k%100==0:print('SOURCE AUDIT',k,flush=True)
 assert maxR<1e-10 and maxc<1e-8 and maxstat<1e-6
 write(OUT/'source_audit.json',dict(passed=True,frames=sum(bool(e['targets']) for e in inv['entries']),candidates=count,independent_quaternion_max=maxR,independent_RGB_position_max_m=maxc,independent_depth_statistics_max_m=maxstat))
 print(read(OUT/'source_audit.json'),flush=True)
if __name__=='__main__':main()
