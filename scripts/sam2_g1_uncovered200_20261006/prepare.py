"""Calibrated RGB/pose preparation for the locked 200-window annotation batch."""
from pathlib import Path
import sys,time,json
import numpy as np,pandas as pd,cv2
from scipy.spatial.transform import Rotation
from projectaria_tools.core import data_provider,calibration
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/sam2_g1_uncovered200_20261006'
sys.path.insert(0,str(ROOT/'scripts/midterm_20260927'))
from common import read_json as read,write_json as write,encoder,finish_encoder,digest
def local(p):return Path('/mnt/'+p[0].lower()+p[2:].replace('\\','/')) if ':' in p else Path(p)
def main():
    cache={};begin=time.monotonic();records=[]
    for c in read(OUT/'selected_clips.json'):
        dest=OUT/c['clip']/'input';dest.mkdir(parents=True,exist_ok=True)
        if (dest/'metadata.json').exists():records.append(read(dest/'metadata.json'));continue
        sid=c['id']
        if sid not in cache:
            provider=data_provider.create_vrs_data_provider(str(ROOT/f'data/Aria/raw_vrs/unknown_20260803_{sid}.vrs'))
            cal=provider.get_device_calibration().get_camera_calib('camera-rgb')
            target=calibration.get_linear_camera_calibration(2560,1920,float(cal.get_focal_lengths()[0]),'pinhole',cal.get_transform_device_camera())
            p=target.projection_params();K=np.array([[p[0],0,p[2]],[0,p[1],p[3]],[0,0,1.]])
            df=pd.read_csv(local(c['source_mps']));ts=np.rint(df.tracking_timestamp_us.to_numpy()*1000).astype(np.int64)
            cache[sid]=(provider,cal,target,K,df,ts)
        provider,cal,target,K,df,ts=cache[sid];ex=cal.get_transform_device_camera().to_matrix();fm=read(local(c['frame_map']))
        times=np.array([int(r['vrs_device_time_ns']) for r in fm],np.int64);ix=np.searchsorted(ts,times).clip(1,len(ts)-1);ix=np.where(abs(ts[ix]-times)<abs(ts[ix-1]-times),ix,ix-1)
        gaps=abs(ts[ix]-times)/1e6;quality=df.quality_score.to_numpy()[ix];graphs=df.graph_uid.to_numpy()[ix]
        assert max(gaps)<=30 and min(quality)>=.5 and len(set(graphs))==1
        dt=np.diff(times)/1e9;assert min(dt)>.08 and max(dt)<.12
        rd=Rotation.from_quat(df[['qx_world_device','qy_world_device','qz_world_device','qw_world_device']].to_numpy()[ix]).as_matrix()
        R=rd@ex[:3,:3];pos=df[['tx_world_device','ty_world_device','tz_world_device']].to_numpy()[ix]+np.einsum('nij,j->ni',rd,ex[:3,3])
        np.savez_compressed(dest/'reference.npz',R=R,K=K,times_ns=times,position=pos,T_device_camera=ex)
        write(dest/'alignment.json',[dict(status='valid',source_frame=r['source_frame'],graph=str(graphs[j]),pose_row=int(ix[j]),nearest_gap_ms=float(gaps[j]),quality=float(quality[j])) for j,r in enumerate(fm)])
        write(dest/'frame_map.json',fm)
        cap=cv2.VideoCapture(str(local(c['source_video'])));cap.set(cv2.CAP_PROP_POS_FRAMES,c['source_start_frame']);proc,log=encoder(dest/'rgb.mp4',2560,1920,10,18)
        for f in range(len(fm)):
            ok,bgr=cap.read();assert ok
            rgb=calibration.distort_by_calibration(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB),target,cal);image=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR);proc.stdin.write(image.tobytes())
            if f==0:cv2.imwrite(str(dest/'first.jpg'),cv2.resize(image,(640,480)))
        cap.release();finish_encoder(proc,log)
        meta=dict(candidate=c['candidate'],clip=c['clip'],source=c,frames=len(fm),width=2560,height=1920,pose_held_frames=0,
          geometry='Factory-calibrated 2560x1920 pinhole RGB; offline Aria reference rotations. No invalid pose holds.',hashes={p.name:digest(p) for p in [dest/'rgb.mp4',dest/'reference.npz',dest/'frame_map.json']})
        write(dest/'metadata.json',meta);records.append(meta);print(c['clip'],'READY',len(fm),'elapsed',round(time.monotonic()-begin,1),flush=True)
    write(OUT/'input_manifest.json',records)
if __name__=='__main__':main()
