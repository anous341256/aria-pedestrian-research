from assoc_common import *
import pandas as pd
from scipy.spatial.transform import Rotation
def main():
 inv=read(OUT/'inventory.json');protected=read(OUT/'source_hashes.json');maxage=0;count=0
 for rec in sorted({c['recording'] for c in inv['clips']}):
  path=ROOT/f'data/Aria/raw_vrs/mps_{rec}_vrs/slam/closed_loop_trajectory.csv';protected[str(path.relative_to(ROOT))]=sha(path)
  cols=['tracking_timestamp_us','graph_uid','quality_score']+[f't{a}_world_device' for a in 'xyz']+[f'q{a}_world_device' for a in 'xyzw'];df=pd.read_csv(path,usecols=cols);ns=df.tracking_timestamp_us.to_numpy(np.int64)*1000;assert (np.diff(ns)>0).all()
  for c in inv['clips']:
   if c['recording']!=rec:continue
   cid=c['clip'];ref=dict(np.load(ROOT/f'outputs/sam2_g1_v6_v8_20261001/{cid}/input/reference.npz'));ts=ref['times_ns'];ii=np.searchsorted(ns,ts,side='right')-1;assert (ii>=0).all();age=ts-ns[ii];assert ((age>=0)&(age<=30000000)).all();assert (df.quality_score.to_numpy()[ii]>=.5).all();assert len(set(df.graph_uid.to_numpy()[ii]))==1
   rd=Rotation.from_quat(df[[f'q{a}_world_device' for a in 'xyzw']].to_numpy()[ii]).as_matrix();ext=ref['T_device_camera'];R=rd@ext[:3,:3];pos=df[[f't{a}_world_device' for a in 'xyz']].to_numpy()[ii]+rd@ext[:3,3];origin=pos[0].copy();pos-=origin
   p=OUT/'poses'/f'{cid}.npz';p.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(p,R=R,c=pos,K=ref['K'],times_ns=ts,selected_mps_ns=ns[ii],world_origin=origin)
   for f in range(len(ts)):assert np.searchsorted(ns[ns<=ts[f]],ts[f],side='right')-1==ii[f]
   maxage=max(maxage,float(age.max()/1e6));count+=len(ts)
 write(OUT/'source_hashes.json',protected);write(OUT/'pose_complete.json',dict(passed=True,frames=count,max_selected_age_ms=maxage,all_selected_at_or_before_RGB=True,all_current_pose_prefix_replays=True))
 print(read(OUT/'pose_complete.json'))
if __name__=='__main__':main()
