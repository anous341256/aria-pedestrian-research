"""Independent geometry/statistics verification; no training-model imports."""
from depth_common import *
import cv2
from scipy.spatial.transform import Rotation
def check_depth():
 plan=read(OUT/'inventory.json');measure={};maxstat=0.;targets=0
 for e in plan['entries']:
  dest=OUT/'frames'/e['key'];done=read(dest/'complete.json');raw=ROOT/done['raw_path'];assert sha(raw)==done['raw_sha256'];a=dict(np.load(raw));d=a['disparity_px'];y,x=np.indices(d.shape)
  right=x-d;xi=np.clip(np.nan_to_num(right),0,511).astype(int);support=a['support_left']&a['support_right'][y,xi]&(right>=0)&(right<=511)
  z=(float(a['baseline_m'])*float(a['intrinsics'][0])/(d+1e-6)).astype(np.float32);valid=support&a['lr_consistent']&np.isfinite(z)&(z>0)&np.isfinite(d)&(d>0)
  yy,xx=np.nonzero(support&np.isfinite(z)&(z>0));zz=z[yy,xx];fx,fy,cx,cy=a['intrinsics'];p=np.column_stack(((xx-cx)*zz/fx,(yy-cy)*zz/fy,zz,np.ones(len(zz))))@a['T_rgb_cv'].T;h=p[:,:3]@a['K_rgb'].T;uv=h[:,:2]/h[:,2,None]
  inside=(p[:,2]>0)&np.isfinite(uv).all(1)&(uv[:,0]>=0)&(uv[:,0]<2559.5)&(uv[:,1]>=0)&(uv[:,1]<1919.5)
  xx=xx[inside];yy=yy[inside];p=p[inside];uv=np.rint(uv[inside]).astype(int);order=np.argsort(p[:,2],kind='stable');_,ii=np.unique(uv[order,1]*2560+uv[order,0],return_index=True);ii=order[ii];uv=uv[ii];p=p[ii];vv=valid[yy[ii],xx[ii]]
  actual={r['target_key']:r for r in read(dest/'measurements.json')};measure[e['key']]=actual
  for t in e['targets']:
   r=t['rle'];counts=r[:-4];x0,y0,x1,y1=map(int,r[-4:]);mask=np.zeros((1920,2560),np.uint8);mask[y0:y1+1,x0:x1+1]=np.repeat(np.arange(len(counts))%2,np.asarray(counts,int)).reshape(y1-y0+1,x1-x0+1);inner=cv2.erode(mask,np.ones((15,15),np.uint8));member=inner[uv[:,1],uv[:,0]].astype(bool);vals=p[member&vv,2];n=len(vals);r=actual[t['target_key']];assert n==r['points'] and int(member.sum())==r['supported_points'];q=n/max(1,int(member.sum()));assert q==r['lr_fraction']
   good=False
   if n>=30:
    med=float(np.median(vals));iqr=float(np.diff(np.quantile(vals,[.25,.75]))[0]);maxstat=max(maxstat,abs(med-r['z_median_m']),abs(iqr-r['z_iqr_m']));good=.5<=med<=30 and q>=.7 and iqr/med<=.25
   assert good==r['usable'];targets+=1
  time=done['timing'];ts=e['rgb_timestamp_ns'];assert time['rgb_timestamp_ns']==ts and max(time['left_timestamp_ns'],time['right_timestamp_ns'])<=ts
  assert ts-time['left_timestamp_ns']<=40000000 and abs(time['left_timestamp_ns']-time['right_timestamp_ns'])<=1000000
 features=np.load(OUT/'depth_features.npz')['features'];expected=np.zeros_like(features);meta=read(BASE/'samples.json')
 for s in plan['sample_refs']:
  last=None
  for j,(key,tk) in enumerate(zip(s['frame_keys'],s['target_keys'])):
   assert int(key.rsplit('_',1)[1])==meta[s['index']]['timestamps_ns'][j]<=meta[s['index']]['timestamp_ns'];r=measure[key][tk]
   if r['usable']:
    z=r['z_median_m'];expected[s['index'],j]=[np.log(z)/np.log(30),np.clip(np.log(z/last),-1,1) if last is not None else 0,1,int(last is not None),r['lr_fraction'],min(r['z_iqr_m']/z,1)];last=z
   else:last=None
 assert np.max(abs(expected-features))<1e-7 and maxstat<1e-6
 return dict(depth_frames=len(plan['entries']),unique_target_frames=targets,depth_stat_max_m=maxstat,feature_max_difference=float(np.max(abs(expected-features))),historical_only=True)
def check_predictions():
 data=dict(np.load(BASE/'dataset.npz'));meta=read(BASE/'samples.json');errmax=0.;geomax=0.;n=0
 for fold in read(BASE/'folds.json'):
  groups=[{meta[i]['recording'] for i in fold[k]} for k in ['train_indices','val_indices','test_indices']];assert not(groups[0]&groups[1] or groups[0]&groups[2] or groups[1]&groups[2]);idx=np.array(fold['test_indices']);K=data['K'][idx]
  for seed in [17,29,43]:
   for arm in ['no_depth','with_depth']:
    p=np.load(OUT/arm/f'fold_{fold["fold"]}/seed_{seed}/predictions.npz');assert np.array_equal(p['indices'],idx)
    r=Rotation.from_rotvec(p['rotation'].astype(float).reshape(-1,3)).as_matrix().reshape(-1,5,3,3);direction=np.einsum('nij,ntj->nti',np.linalg.inv(K),np.concatenate([p['component'],np.ones((len(idx),5,1))],-1));rays=np.einsum('ntji,ntj->nti',r,direction);h=np.einsum('nij,ntj->nti',K,rays);xy=h[...,:2]/h[...,2,None];error=np.linalg.norm(xy-data['xy'][idx,16:21],axis=-1);assert (h[...,2]>1e-6).all();geomax=max(geomax,float(np.max(abs(xy-p['image']))));errmax=max(errmax,float(np.max(abs(error-p['image_error']))));n+=len(idx)*5
 assert errmax<1e-6 and geomax<1e-6
 return dict(independent_future_points=n,projection_max_px=geomax,error_max_px=errmax,recording_splits_disjoint=True)
if __name__=='__main__':
 result=check_depth();result.update(check_predictions());result['passed']=True;write(OUT/'independent_audit.json',result);print(result)
