"""Extend reference camera positions to future labels, without changing history."""
from geometry_common import *
import pandas as pd
from scipy.spatial.transform import Rotation
def main():
 d=dict(np.load(BASE/'dataset.npz'));c=dict(np.load(BASE/'camera_dataset.npz'));meta=read(BASE/'samples.json');names=read(CAM/'inventory.json')['record_names'];cal=read(CAM/'calibrations.json');protected=read(OUT/'source_hashes.json');cb=np.empty((len(c['rotation']),21,3));ctimes=np.empty((len(cb),21),np.int64);cselected=np.empty_like(ctimes);pb=np.empty((117,21,3));pworld=np.empty_like(pb);psel=np.empty((117,21),np.int64);boundary=0;maxage=0.
 for rec,name in enumerate(names):
  source=ROOT/f'data/Aria/raw_vrs/mps_{name}_vrs/slam/closed_loop_trajectory.csv';framepath=CAM/f'poses/{name}.npz';protected[str(source.relative_to(ROOT))]=sha(source);protected[str(framepath.relative_to(ROOT))]=sha(framepath)
  df=pd.read_csv(source,usecols=['tracking_timestamp_us','graph_uid','quality_score']+[f't{a}_world_device' for a in 'xyz']+[f'q{a}_world_device' for a in 'xyzw']);ns=df.tracking_timestamp_us.to_numpy(np.int64)*1000;frames=dict(np.load(framepath));ts=frames['times_ns'];assert (np.diff(ns)>0).all();ii=(np.searchsorted(ns,ts,side='right')-1).clip(0);age=ts-ns[ii];q=df[[f'q{a}_world_device' for a in 'xyzw']].to_numpy()[ii];pos=df[[f't{a}_world_device' for a in 'xyz']].to_numpy()[ii];rd=Rotation.from_quat(q).as_matrix();ext=np.asarray(cal[name]['T_device_camera']);xyz=pos+rd@ext[:3,3];quality=df.quality_score.to_numpy()[ii];graph=df.graph_uid.to_numpy()[ii]
  selected=np.flatnonzero(c['record_index']==rec);fs=c['frame_index'][selected,None]+np.arange(-15,6);assert ((fs>=0)&(fs<len(ts))).all();assert (abs(age[fs])<=30000000).all() and (quality[fs]>=.5).all();assert (graph[fs]==graph[fs[:,15,None]]).all();assert (ns[ii[fs[:,:16]]]<=ts[fs[:,15,None]]).all();boundary+=int((age[fs[:,:16]]<0).sum());maxage=max(maxage,float(age[fs].max()/1e6));cb[selected]=(xyz[fs]-xyz[fs[:,15,None]])@frames['R'][fs[:,15]];ctimes[selected]=ts[fs];cselected[selected]=ns[ii[fs]]
  for j,m in enumerate(meta):
   if m['recording']!=name:continue
   tt=np.array(m['timestamps_ns']);ff=np.searchsorted(ts,tt);assert np.array_equal(ts[ff],tt);assert (quality[ff]>=.5).all() and (abs(age[ff])<=30000000).all();assert (graph[ff]==graph[ff[15]]).all();assert (ns[ii[ff[:16]]]<=tt[15]).all();pb[j]=(xyz[ff]-xyz[ff[15]])@d['R'][j,15];pworld[j]=xyz[ff];psel[j]=ns[ii[ff]]
  print('MPS',name,len(selected),flush=True)
 histcam=float(np.max(abs(cb[:,:16]-c['translation'])));histperson=float(np.max(abs(pb[:,:16]-d['translation'])));assert histcam<1e-6 and histperson<1e-6
 d['translation']=pb;d['translation_world_position']=pworld;d['translation_mps_ns']=psel;d['A']=np.swapaxes(d['R'][:,15],-1,-2)[:,None]@d['R'];c['translation']=cb.astype(np.float32);c['translation_rgb_ns']=ctimes;c['translation_mps_ns']=cselected;c['seconds']=(ctimes-ctimes[:,15,None])/1e9
 np.savez_compressed(OUT/'person_poses.npz',**d);np.savez_compressed(OUT/'camera_dataset.npz',**c);write(OUT/'source_hashes.json',protected);write(OUT/'pose_preparation.json',dict(passed=True,camera_windows=len(cb),historical_start_boundary_occurrences=boundary,max_selected_age_ms=maxage,history_camera_max_m=histcam,history_person_max_m=histperson))
def combine():
 d=dict(np.load(OUT/'person_poses.npz'));depth=dict(np.load(OUT/'depth_labels.npz'));d.update(depth=depth['z'],depth_valid=depth['valid'],depth_quality=depth['quality']);xc=lift(d['xy'],d['depth'],d['K']);xyz=to_fixed(xc,d['A'],d['translation']);xyz[~d['depth_valid']]=0;d['xyz']=xyz
 assert d['depth_valid'][:,:16].any(1).all();back,z=project(to_camera(xyz,d['A'],d['translation']),d['K']);error=float(np.max(abs(back[d['depth_valid']]-d['xy'][d['depth_valid']])));assert error<1e-8
 np.savez_compressed(OUT/'dataset.npz',**d);write(OUT/'geometry_preflight.json',dict(passed=True,real_roundtrip_max_px=error,valid_future_labels=int(d['depth_valid'][:,16:].sum()),total_future_labels=585))
def synthetic():
 rng=np.random.default_rng(21);A=Rotation.from_rotvec(rng.normal(0,.15,(120,3))).as_matrix().reshape(4,30,3,3);b=rng.normal(0,.2,(4,30,3));K=np.tile([[1000.,0,1280],[0,1000,960],[0,0,1]],(4,1,1));p=rng.normal(0,.3,(4,30,3));p[:,:,2]+=5
 xc=to_camera(p,A,b);uv,z=project(xc,K);roundtrip=to_fixed(lift(uv,z,K),A,b);err=float(np.max(abs(roundtrip-p)));assert err<1e-10
 static=np.repeat(p[:,:1],30,1);x=to_camera(static,A,b);uv,z=project(x,K);fixed=to_fixed(lift(uv,z,K),A,b);staticerr=float(np.max(abs(fixed-static)));assert staticerr<1e-10
 identity=np.broadcast_to(np.eye(3),(1,1,3,3));bp=np.array([[[.1,0,0]]]);world=np.array([[[0,0,2.]]]);pixel,_=project(to_camera(world,identity,bp),K[:1]);assert abs(pixel[0,0,0]-(1280-50))<1e-9
 write(OUT/'synthetic_geometry.json',dict(passed=True,random_roundtrip_max_m=err,static_scene_compensation_max_m=staticerr,right_translation_pixel_shift=float(pixel[0,0,0]-1280)))
if __name__=='__main__':
 import sys
 if len(sys.argv)>1 and sys.argv[1]=='combine':combine();synthetic()
 else:main();synthetic()
