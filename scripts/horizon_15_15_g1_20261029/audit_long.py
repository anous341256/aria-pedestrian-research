"""Independent data, pose, split and projection checks (SciPy runtime)."""
from pathlib import Path
import json,hashlib,collections
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/horizon_15_15_g1_20261029';BULK=ROOT/'outputs/g1_bulk_association_20261029';OLD=ROOT/'outputs/translation_forecast_20261020';CAM=ROOT/'outputs/camera_full_data_20261002'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def quaternion(q):
 x,y,z,w=q.T;n=x*x+y*y+z*z+w*w;s=2/n
 return np.array([1-s*(y*y+z*z),s*(x*y-z*w),s*(x*z+y*w),s*(x*y+z*w),1-s*(x*x+z*z),s*(y*z-x*w),s*(x*z-y*w),s*(y*z+x*w),1-s*(x*x+y*y)]).T.reshape(-1,3,3)
def main():
 d=dict(np.load(OUT/'dataset.npz'));meta=read(OUT/'samples.json');cfg=read(OUT/'protocol.json');folds=read(OUT/'folds.json');od=dict(np.load(OLD/'dataset.npz'));om=read(OLD/'samples.json');lookup={}
 for i,m in enumerate(om):
  for j,t in enumerate(m['timestamps_ns']):lookup[(m['clip'],m['id'],m['segment'],t)]=(od['xy'][i,j],od['boxes'][i,j],od['R'][i,j],od['K'][i])
 inv=read(BULK/'inventory.json');bulk={t['target_key']:(e,t) for e in inv['entries'] for t in e['targets']};posecache={};maxround=0.;maxtr=0.;oldcount=0;newcount=0
 for i,m in enumerate(meta):
  ts=np.array(m['timestamps_ns']);assert len(ts)==31 and ((np.diff(ts)>90e6)&(np.diff(ts)<110e6)).all();assert np.allclose(d['seconds'][i],(ts-ts[15])/1e9,rtol=0,atol=1e-14)
  if i<cfg['original_windows']:
   for j,t in enumerate(ts):
    x,b,R,K=lookup[(m['clip'],m['id'],m['segment'],int(t))];assert np.array_equal(x,d['xy'][i,j]) and np.array_equal(b,d['boxes'][i,j]) and np.array_equal(R,d['R'][i,j]) and np.array_equal(K,d['K'][i]);oldcount+=1
  else:
   for j,u in enumerate(m['uids']):
    e,t=bulk[u];b=np.array(t['box']);assert np.array_equal(b,d['boxes'][i,j]);assert np.array_equal((b[:2]+b[2:])/2,d['xy'][i,j]);assert ts[j]==e['rgb_timestamp_ns'];cid=e['clip']
    if cid not in posecache:posecache[cid]=dict(np.load(BULK/'poses'/f'{cid}.npz'))
    assert np.array_equal(posecache[cid]['R'][e['frame']],d['R'][i,j]);newcount+=1
   R=d['R'][i];ang=np.degrees(Rotation.from_matrix(np.swapaxes(R[:-10],1,2)@R[10:]).magnitude());speed=np.degrees(Rotation.from_matrix(np.swapaxes(R[:-1],1,2)@R[1:]).magnitude())/(np.diff(ts)/1e9);assert ang.max()<=20 and np.percentile(speed,95)<=50
  K=d['K'][i];R0=d['R'][i,15]
  for j in range(31):
   ray=np.linalg.solve(K,np.r_[d['xy'][i,j],1.]);q=K@(R0.T@(d['R'][i,j]@ray));p=q[:2]/q[2];h=d['H'][i,j]@np.r_[d['xy'][i,j],1.];assert np.max(abs(p-h[:2]/h[2]))<1e-8
   back=K@(d['R'][i,j].T@(R0@np.linalg.solve(K,np.r_[p,1.])));maxround=max(maxround,float(abs(back[:2]/back[2]-d['xy'][i,j]).max()))
  assert (d['translation_mps_ns'][i]<=ts[15]).all()
 cd=dict(np.load(OUT/'camera_dataset.npz'));names=read(OUT/'camera_record_names.json');cal=read(CAM/'calibrations.json');maxcam=0.;camera_histories=0
 for name in names:
  cs=['tracking_timestamp_us']+[f't{x}_world_device' for x in 'xyz']+[f'q{x}_world_device' for x in 'xyzw'];df=pd.read_csv(ROOT/f'data/Aria/raw_vrs/mps_{name}_vrs/slam/closed_loop_trajectory.csv',usecols=cs);ns=df.tracking_timestamp_us.to_numpy(np.int64)*1000;pose=dict(np.load(CAM/f'poses/{name}.npz'));ii=np.searchsorted(ns,pose['times_ns'],side='right')-1;ii=ii.clip(0);rd=quaternion(df[[f'q{x}_world_device' for x in 'xyzw']].to_numpy()[ii]);ext=np.array(cal[name]['T_device_camera']);pos=df[[f't{x}_world_device' for x in 'xyz']].to_numpy()[ii]+np.einsum('nij,j->ni',rd,ext[:3,3]);ci=np.flatnonzero(cd['record_index']==names.index(name));fs=cd['start_frame'][ci,None]+np.arange(31);rel=np.einsum('nji,ntjk->ntik',pose['R'][fs[:,15]],pose['R'][fs]);maxcam=max(maxcam,float(abs(Rotation.from_rotvec(cd['rotation'][ci].reshape(-1,3)).as_matrix().reshape(-1,31,3,3)-rel).max()));expected=np.einsum('nji,ntj->nti',pose['R'][fs[:,15]],pos[fs[:,:16]]-pos[fs[:,15,None]]);maxtr=max(maxtr,float(abs(expected-cd['translation'][ci]).max()));assert np.array_equal(ns[ii[fs[:,:16]]],cd['translation_mps_ns'][ci]);camera_histories+=len(ci)
  for i,m in enumerate(meta):
   if m['recording']!=name:continue
   ts=np.array(m['timestamps_ns'][:16]);f=np.searchsorted(pose['times_ns'],ts);assert np.array_equal(pose['times_ns'][f],ts);tr=np.einsum('ji,tj->ti',d['R'][i,15],pos[f]-pos[f[-1]]);maxtr=max(maxtr,float(abs(tr-d['translation'][i]).max()));assert np.array_equal(ns[ii[f]],d['translation_mps_ns'][i])
 for f in folds:
  held={f['test'],f['validation']};assert all(meta[i]['recording'] not in held for i in f['expanded_train_indices']);assert all(names[cd['record_index'][i]] not in held for i in f['camera']['train_indices']);assert {meta[i]['recording'] for i in f['test_indices']}=={f['test']};assert {meta[i]['recording'] for i in f['val_indices']}=={f['validation']}
 assert maxtr<1e-6 and maxcam<1e-6 and maxround<1e-8
 result=dict(passed=True,old_target_frames=oldcount,new_target_frames=newcount,camera_histories=camera_histories,manual_quaternion_translation_max_m=maxtr,rotation_cache_max=maxcam,projection_roundtrip_max_px=maxround,recording_disjoint=True,all_new_windows_pass_boundary_camera_screen=True)
 if (OUT/'training_complete.json').exists():
  rows=read(OUT/'metrics.json');maxe=0.;maxproj=0.;n=0
  for f in folds:
   for s in cfg['seeds']:
    for arm in cfg['arms']:
     z=dict(np.load(OUT/arm/f'fold_{f["fold"]}/seed_{s}/predictions.npz'));idx=z['indices'];e=np.linalg.norm(z['image']-d['xy'][idx,16:31],axis=-1);assert (z['image_depth']>1e-6).all();maxe=max(maxe,float(abs(e-z['image_error']).max()))
     for j,i in enumerate(idx):
      rr=next(r for r in rows if r['index']==i and r['seed']==s and r['arm']==arm);maxe=max(maxe,abs(rr['ADE']-e[j].mean()),abs(rr['FDE']-e[j,-1]));n+=1;ray=np.linalg.solve(d['K'][i],np.c_[z['component'][j],np.ones(15)].T).T;R=Rotation.from_rotvec(z['rotation'][j]).as_matrix();q=(d['K'][i]@np.einsum('tji,tj->ti',R,ray).T).T;p=q[:,:2]/q[:,2,None];maxproj=max(maxproj,float(abs(p-z['image'][j]).max()))
  assert maxe<1e-8 and maxproj<1e-8;result.update(prediction_rows=n,metric_max_px=maxe,scipy_projection_max_px=maxproj)
 write(OUT/'independent_audit.json',result);print(result,flush=True)
if __name__=='__main__':main()
