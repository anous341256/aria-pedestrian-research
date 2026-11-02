"""Prepare exactly the selected export frames in a calibrated pinhole domain."""
import sys,json
from pathlib import Path
import cv2,numpy as np,pandas as pd
from scipy.spatial.transform import Rotation
from projectaria_tools.core import data_provider,calibration
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/midterm_20260927'))
from common import read_json,write_json,encoder,finish_encoder,digest
OUT=ROOT/'outputs/sam2_g1_factor_supplement_20261006'
def posix(s):return Path('/mnt/'+s[0].lower()+s[2:].replace('\\','/'))

def run():
    selected=read_json(ROOT/'outputs/sam2_g1_factor_supplement_20261006/selected_clips.json')
    records=[]
    for c in selected:
        out=OUT/f"{c['candidate']:02d}"/'input';out.mkdir(parents=True,exist_ok=True)
        if (out/'metadata.json').exists():
            records.append(read_json(out/'metadata.json'));continue
        provider=data_provider.create_vrs_data_provider(str(ROOT/f"data/Aria/raw_vrs/unknown_20260803_{c['id']}.vrs"))
        calib=provider.get_device_calibration().get_camera_calib('camera-rgb')
        w,h=map(int,calib.get_image_size());assert (w,h)==(2560,1920)
        target=calibration.get_linear_camera_calibration(w,h,float(calib.get_focal_lengths()[0]),'pinhole',calib.get_transform_device_camera())
        p=target.projection_params();K=np.array([[p[0],0,p[2]],[0,p[1],p[3]],[0,0,1]])
        extr=calib.get_transform_device_camera().to_matrix()
        frames=read_json(posix(c['frame_map']));times=np.array([int(x['vrs_device_time_ns']) for x in frames],dtype=np.int64)
        assert np.all(np.diff(times)>0),'Duplicate or reversed timestamps require explicit investigation.'
        df=pd.read_csv(posix(c['source_mps']))
        ts=np.rint(df.tracking_timestamp_us.to_numpy()*1000).astype(np.int64)
        idx=np.searchsorted(ts,times);idx=np.clip(idx,1,len(ts)-1)
        idx=np.where(abs(ts[idx]-times)<abs(ts[idx-1]-times),idx,idx-1)
        q=df[['qx_world_device','qy_world_device','qz_world_device','qw_world_device']].to_numpy()
        R=[];alignment=[];held=0
        for f,j in enumerate(idx):
            gap=float(abs(int(ts[j])-int(times[f]))/1e6);quality=float(df.quality_score.iloc[j])
            valid=gap<=30 and quality>=.5 and np.isfinite(q[j]).all()
            assert valid or f>0,'Initial reference unavailable.'
            R.append(Rotation.from_quat(q[j]).as_matrix()@extr[:3,:3] if valid else R[-1].copy())
            held+=int(not valid)
            alignment.append(dict(status='valid' if valid else 'held_previous_valid',graph=str(df.graph_uid.iloc[j]),pose_row=int(j),nearest_gap_ms=gap,quality=quality,source_frame=frames[f]['source_frame']))
        assert len({a['graph'] for a in alignment})==1
        np.savez_compressed(out/'reference.npz',R=np.array(R),K=K,times_ns=times,T_device_camera=extr)
        write_json(out/'alignment.json',alignment);write_json(out/'frame_map.json',frames)
        cap=cv2.VideoCapture(str(posix(c['source_video'])));cap.set(cv2.CAP_PROP_POS_FRAMES,c['source_start_frame'])
        proc,log=encoder(out/'rgb.mp4',w,h,10,18)
        for f in range(len(frames)):
            ok,bgr=cap.read();assert ok
            rgb=calibration.distort_by_calibration(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB),target,calib)
            image=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
            proc.stdin.write(image.tobytes())
            if f==0:cv2.imwrite(str(out/'first.jpg'),cv2.resize(image,(1280,960)),[cv2.IMWRITE_JPEG_QUALITY,95])
        finish_encoder(proc,log);cap.release()
        meta=dict(candidate=c['candidate'],source=c,frames=len(frames),width=w,height=h,pose_held_frames=held,
                  pose_method='Nearest offline Aria MPS reference within 30ms and quality>=0.5; otherwise hold previous valid orientation, shared by both methods.',
                  geometry='Original exported fisheye RGB rectified with factory calibration to 2560x1920 pinhole; rotation only.',
                  hashes={p.name:digest(p) for p in [out/'rgb.mp4',out/'reference.npz',out/'frame_map.json']})
        write_json(out/'metadata.json',meta);records.append(meta)
        print(f"Prepared {c['candidate']:02d}: {len(frames)} frames, {held} held poses",flush=True)
    write_json(OUT/'input_manifest.json',records)
if __name__=='__main__':run()
