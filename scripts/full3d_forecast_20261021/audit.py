"""Independent row-vector/world-coordinate and raw quaternion checks."""
from geometry_common import *
import cv2,pandas as pd
from scipy.spatial.transform import Rotation
def quat(q):
 q=q/np.linalg.norm(q,axis=1,keepdims=True);x,y,z,w=q.T
 return np.stack([1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w),2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w),2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)],-1).reshape(-1,3,3)
def sources():
 d=dict(np.load(OUT/'dataset.npz'));c=dict(np.load(OUT/'camera_dataset.npz'));meta=read(BASE/'samples.json');names=read(CAM/'inventory.json')['record_names'];cal=read(CAM/'calibrations.json');maxcam=0.;maxperson=0.;prefix=0;count=0
 for rec,name in enumerate(names):
  df=pd.read_csv(ROOT/f'data/Aria/raw_vrs/mps_{name}_vrs/slam/closed_loop_trajectory.csv',usecols=['tracking_timestamp_us']+[f't{a}_world_device' for a in 'xyz']+[f'q{a}_world_device' for a in 'xyzw']);ns=df.tracking_timestamp_us.to_numpy(np.int64)*1000;rd=quat(df[[f'q{a}_world_device' for a in 'xyzw']].to_numpy());world=df[[f't{a}_world_device' for a in 'xyz']].to_numpy()+rd@np.asarray(cal[name]['T_device_camera'])[:3,3];frames=dict(np.load(CAM/f'poses/{name}.npz'));ix=np.flatnonzero(c['record_index']==rec);ts=c['translation_rgb_ns'][ix];j=(np.searchsorted(ns,ts,side='right')-1).clip(0);assert np.array_equal(ns[j],c['translation_mps_ns'][ix]);assert (ns[j[:,:16]]<=ts[:,15,None]).all();expected=(world[j]-world[j[:,15,None]])@frames['R'][c['frame_index'][ix]];maxcam=max(maxcam,float(np.max(abs(expected-c['translation'][ix]))));count+=len(ix)*21
  for k,m in enumerate(meta):
   if m['recording']!=name:continue
   ts=np.array(m['timestamps_ns']);j=(np.searchsorted(ns,ts,side='right')-1).clip(0);expected=(world[j]-world[j[15]])@d['R'][k,15];maxperson=max(maxperson,float(np.max(abs(expected-d['translation'][k]))));assert np.array_equal(ns[j],d['translation_mps_ns'][k]);limited=ns[ns<=ts[15]];assert np.array_equal((np.searchsorted(limited,ts[:16],side='right')-1).clip(0),j[:16]);prefix+=1
 assert maxcam<1e-6 and maxperson<1e-8
 # Reconstruct XYZ by explicitly going through world coordinates (no shared geometry helpers).
 uvh=np.concatenate([d['xy'],np.ones((117,21,1))],-1);rays=uvh@np.swapaxes(np.linalg.inv(d['K']),-1,-2);xc=rays*d['depth'][...,None];world=np.einsum('ntij,ntj->nti',d['R'],xc)+d['translation_world_position'];fixed=np.einsum('nji,ntj->nti',d['R'][:,15],world-d['translation_world_position'][:,15,None]);valid=d['depth_valid'];err=float(np.max(abs(fixed[valid]-d['xyz'][valid])));assert err<1e-8
 cam=(fixed-d['translation'])[:,:,:,None];# separate inverse via row-vector R_t.T R_0
 back=np.einsum('ntji,ntj->nti',d['R'],np.einsum('nij,ntj->nti',d['R'][:,15],fixed)+d['translation_world_position'][:,15,None]-d['translation_world_position']);h=back@np.swapaxes(d['K'],-1,-2);pixel=h[...,:2]/np.where(abs(h[...,2:])>1e-12,h[...,2:],1e-12);pixelerr=float(np.max(abs(pixel[valid]-d['xy'][valid])));assert pixelerr<1e-8
 write(OUT/'source_geometry_audit.json',dict(passed=True,raw_MPS_positions=count,manual_quaternion_camera_max_m=maxcam,manual_quaternion_person_max_m=maxperson,historical_raw_MPS_prefix_checks=prefix,independent_world_route_xyz_max_m=err,independent_world_roundtrip_max_px=pixelerr))
def depth():
 plan=read(OUT/'inventory.json');labels=np.load(OUT/'depth_labels.npz');allmeasure={};maxstat=0.;targets=0
 for e in plan['entries']:
  folder=OUT/'frames'/e['key'];done=read(folder/'complete.json');raw=ROOT/done['raw_path'];assert sha(raw)==done['raw_sha256'];a=dict(np.load(raw));disp=a['disparity_px'];y,x=np.indices(disp.shape);xr=x-disp;xi=np.clip(np.nan_to_num(xr),0,511).astype(int);support=a['support_left']&a['support_right'][y,xi]&(xr>=0)&(xr<=511);z=(float(a['baseline_m'])*float(a['intrinsics'][0])/(disp+1e-6)).astype(np.float32);valid=support&a['lr_consistent']&np.isfinite(z)&(z>0)&np.isfinite(disp)&(disp>0)
  yy,xx=np.nonzero(support&np.isfinite(z)&(z>0));zz=z[yy,xx];fx,fy,cx,cy=a['intrinsics'];p=np.column_stack(((xx-cx)*zz/fx,(yy-cy)*zz/fy,zz,np.ones(len(zz))))@a['T_rgb_cv'].T;h=p[:,:3]@a['K_rgb'].T;uv=h[:,:2]/h[:,2,None];inside=(p[:,2]>0)&np.isfinite(uv).all(1)&(uv[:,0]>=0)&(uv[:,0]<2559.5)&(uv[:,1]>=0)&(uv[:,1]<1919.5);xx=xx[inside];yy=yy[inside];p=p[inside];uv=np.rint(uv[inside]).astype(int);order=np.argsort(p[:,2],kind='stable');_,ii=np.unique(uv[order,1]*2560+uv[order,0],return_index=True);ii=order[ii];p=p[ii];uv=uv[ii];vv=valid[yy[ii],xx[ii]];actual={r['target_key']:r for r in read(folder/'measurements.json')};allmeasure[e['key']]=actual
  for t in e['targets']:
   r=t['rle'];x0,y0,x1,y1=map(int,r[-4:]);mask=np.zeros((1920,2560),np.uint8);mask[y0:y1+1,x0:x1+1]=np.repeat(np.arange(len(r[:-4]))%2,np.asarray(r[:-4],int)).reshape(y1-y0+1,x1-x0+1);mask=cv2.erode(mask,np.ones((15,15),np.uint8));member=mask[uv[:,1],uv[:,0]].astype(bool);vals=p[member&vv,2];n=len(vals);r=actual[t['target_key']];assert n==r['points'] and int(member.sum())==r['supported_points'];q=n/max(1,int(member.sum()));assert q==r['lr_fraction'];usable=False
   if n>=30:
    med=float(np.median(vals));iqr=float(np.diff(np.quantile(vals,[.25,.75]))[0]);maxstat=max(maxstat,abs(med-r['z_median_m']),abs(iqr-r['z_iqr_m']));usable=.5<=med<=30 and q>=.7 and iqr/med<=.25
   assert usable==r['usable'];targets+=1
  time=done['timing'];assert time['rgb_timestamp_ns']==e['rgb_timestamp_ns'];assert max(time['left_timestamp_ns'],time['right_timestamp_ns'])<=e['rgb_timestamp_ns'];assert e['rgb_timestamp_ns']-time['left_timestamp_ns']<=40000000
 for s in plan['sample_refs']:
  for j,(key,tk) in enumerate(zip(s['frame_keys'],s['target_keys'])):
   r=allmeasure[key][tk];i=s['index'];assert r['usable']==labels['valid'][i,j]
   if r['usable']:assert r['z_median_m']==labels['z'][i,j]
 assert maxstat<1e-6;write(OUT/'depth_audit.json',dict(passed=True,frames=len(plan['entries']),unique_targets=targets,label_occurrences=117*21,raw_depth_statistics_max_m=maxstat))
def predictions():
 d=dict(np.load(OUT/'dataset.npz'));c=np.load(OUT/'camera_dataset.npz');meta=read(BASE/'samples.json');maxgeo=0.;maxerror=0.;maxangle=0.;n=0;invalid=0
 for fold,cf in zip(read(BASE/'folds.json'),read(BASE/'camera_folds.json')):
  groups=[{meta[i]['recording'] for i in fold[k]} for k in ['train_indices','val_indices','test_indices']];assert not(groups[0]&groups[1] or groups[0]&groups[2] or groups[1]&groups[2]);groups=[set(c['record_index'][cf[k]].tolist()) for k in ['train_indices','val_indices','test_indices']];assert not(groups[0]&groups[1] or groups[0]&groups[2] or groups[1]&groups[2]);idx=np.array(fold['test_indices'])
  for seed in [17,29,43]:
   p=np.load(OUT/f'fold_{fold["fold"]}/seed_{seed}/person/predictions.npz');assert np.array_equal(p['indices'],idx);A=Rotation.from_rotvec(p['rotation'].astype(float).reshape(-1,3)).as_matrix().reshape(-1,5,3,3);xc=np.einsum('ntj,ntjk->ntk',p['person_xyz']-p['translation'],A);h=xc@np.swapaxes(d['K'][idx],-1,-2);uv=h[...,:2]/np.where(abs(h[...,2:])>1e-12,h[...,2:],1e-12);err=np.linalg.norm(uv-d['xy'][idx,16:],axis=-1);bad=(xc[...,2]<=1e-6)|~np.isfinite(err);err[bad]=32000;invalid+=int(bad.sum());maxgeo=max(maxgeo,float(np.max(abs(uv-p['image']))));maxerror=max(maxerror,float(np.max(abs(err-p['image_error']))));angle=np.rad2deg((Rotation.from_rotvec(p['rotation'].astype(float).reshape(-1,3)).inv()*Rotation.from_matrix(d['A'][idx,16:].reshape(-1,3,3))).magnitude()).reshape(-1,5);maxangle=max(maxangle,float(np.max(abs(angle-p['angle_deg']))));n+=len(idx)*5
 assert maxgeo<1e-6 and maxerror<1e-6 and maxangle<1e-5;write(OUT/'prediction_audit.json',dict(passed=True,future_positions=n,independent_projection_max_px=maxgeo,independent_metric_max_px=maxerror,independent_angle_max_deg=maxangle,invalid_rays=invalid,recording_splits_disjoint=True))
if __name__=='__main__':
 import sys
 {'sources':sources,'depth':depth,'predictions':predictions}[sys.argv[1]]()
