"""Independent quaternion/raw disparity audit plus final candidate/index checks."""
from bulk_common import *
import cv2,pandas as pd
cv2.setNumThreads(1)
from audit_sources import quaternion

def main():
 inv=read(OUT/'inventory.json');maxR=maxc=0.;posecount=0
 for rec in sorted({c['recording'] for c in inv['clips']}):
  df=pd.read_csv(ROOT/f'data/Aria/raw_vrs/mps_{rec}_vrs/slam/closed_loop_trajectory.csv',usecols=['tracking_timestamp_us']+[f't{a}_world_device' for a in 'xyz']+[f'q{a}_world_device' for a in 'xyzw']);ns=df.tracking_timestamp_us.to_numpy(np.int64)*1000
  for cl in inv['clips']:
   if cl['recording']!=rec:continue
   p=dict(np.load(OUT/'poses'/f'{cl["clip"]}.npz'));cal=np.load(ROOT/cl['input']/'reference.npz');ext=cal['T_device_camera'];j=np.searchsorted(ns,p['times_ns'],side='right')-1;rd=quaternion(df[[f'q{a}_world_device' for a in 'xyzw']].to_numpy()[j]);pos=df[[f't{a}_world_device' for a in 'xyz']].to_numpy()[j]+rd@ext[:3,3];pos-=pos[0].copy();maxR=max(maxR,float(abs(rd@ext[:3,:3]-p['R']).max()));maxc=max(maxc,float(abs(pos-p['c']).max()));assert np.array_equal(ns[j],p['selected_mps_ns']);assert (ns[j]<=p['times_ns']).all();posecount+=len(j)
 assert maxR<1e-10 and maxc<1e-8;write(OUT/'pose_independent_audit.json',dict(passed=True,frames=posecount,max_rotation_difference=maxR,max_position_difference_m=maxc))
 count=0;maxstat=0.;failed=[];allcenters=0
 for k,e in enumerate(sorted(inv['entries'],key=lambda e:(e['recording'],e['rgb_timestamp_ns']))):
  if not e['targets']:continue
  dest=OUT/'depth'/e['key'];mark=OUT/'audit_frames'/f'{e["key"]}.json'
  if mark.exists():
   z=read(mark);count+=z['targets'];maxstat=max(maxstat,z['maxstat']);allcenters+=z['targets'];continue
  while not (dest/'complete.json').exists() and not (dest/'failure.json').exists():time.sleep(5)
  if not (dest/'complete.json').exists():failed.append(e['key']);continue
  done=read(dest/'complete.json');raw=ROOT/done['raw_path'];assert sha(raw)==done['raw_sha256'];a=dict(np.load(raw));disp=a['disparity_px'];y,x=np.indices(disp.shape);xr=x-disp;xi=np.clip(np.nan_to_num(xr),0,511).astype(int);support=a['support_left']&a['support_right'][y,xi]&(xr>=0)&(xr<=511);z=(float(a['baseline_m'])*float(a['intrinsics'][0])/(disp+1e-6)).astype(np.float32);valid=support&a['lr_consistent']&np.isfinite(z)&(z>0)&np.isfinite(disp)&(disp>0)
  yy,xx=np.nonzero(support&np.isfinite(z)&(z>0));zz=z[yy,xx];fx,fy,cx,cy=a['intrinsics'];p=np.column_stack(((xx-cx)*zz/fx,(yy-cy)*zz/fy,zz,np.ones(len(zz))))@a['T_rgb_cv'].T;h=p[:,:3]@a['K_rgb'].T;uv=h[:,:2]/h[:,2,None];keep=(p[:,2]>0)&np.isfinite(uv).all(1)&(uv[:,0]>=0)&(uv[:,0]<2559.5)&(uv[:,1]>=0)&(uv[:,1]<1919.5);xx=xx[keep];yy=yy[keep];p=p[keep];uv=np.rint(uv[keep]).astype(int);order=np.argsort(p[:,2],kind='stable');_,ii=np.unique(uv[order,1]*2560+uv[order,0],return_index=True);ii=order[ii];p=p[ii];uv=uv[ii];vv=valid[yy[ii],xx[ii]];actual={r['target_key']:r for r in read(dest/'measurements.json')};localmax=0.
  for t in e['targets']:
   r=t['rle'];x0,y0,x1,y1=map(int,r[-4:]);mask=np.zeros((1920,2560),np.uint8);mask[y0:y1+1,x0:x1+1]=np.repeat(np.arange(len(r[:-4]))%2,np.asarray(r[:-4],int)).reshape(y1-y0+1,x1-x0+1);bx,by,bw,bh=cv2.boundingRect(mask);assert np.array_equal(np.array([bx,by,bx+bw,by+bh]),t['box']);allcenters+=1
   mask=cv2.erode(mask,np.ones((15,15),np.uint8));member=mask[uv[:,1],uv[:,0]].astype(bool);vals=p[member&vv,2];n=len(vals);r=actual[t['target_key']];assert n==r['points'] and int(member.sum())==r['supported_points'];q=n/max(1,int(member.sum()));assert q==r['lr_fraction'];usable=False
   if n>=30:
    med=float(np.median(vals));iqr=float(np.diff(np.quantile(vals,[.25,.75]))[0]);localmax=max(localmax,abs(med-r['z_median_m']),abs(iqr-r['z_iqr_m']));usable=.5<=med<=30 and q>=.7 and iqr/med<=.25
   assert usable==r['usable'];count+=1
  timing=done['timing'];assert timing['rgb_timestamp_ns']==e['rgb_timestamp_ns'];assert max(timing['left_timestamp_ns'],timing['right_timestamp_ns'])<=e['rgb_timestamp_ns'];assert timing['left_timestamp_ns']<=e['rgb_timestamp_ns']-1000000;assert abs(timing['left_timestamp_ns']-timing['right_timestamp_ns'])<=1000000;maxstat=max(maxstat,localmax);write(mark,dict(passed=True,targets=len(e['targets']),maxstat=localmax))
  if k%50==0:write(OUT/'source_audit_progress.json',dict(frames_seen=k+1,total=len(inv['entries']),targets=count));print('AUDIT',k+1,count,flush=True)
 assert maxstat<1e-6;write(OUT/'source_audit.json',dict(passed=True,pose_frames=posecount,candidates=count,mask_boxes_checked=allcenters,independent_rotation_max=maxR,independent_position_max_m=maxc,independent_depth_statistics_max_m=maxstat,unavailable_depth_frames=failed))
if __name__=='__main__':main()
