"""Export frozen clips and synchronized MPS references. Run in /opt/aria-vrs."""
import argparse
import csv
import importlib.metadata
import time
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation, Slerp
from projectaria_tools.core import calibration, data_provider, mps
from projectaria_tools.core.sensor_data import TimeDomain
from projectaria_tools.core.mps.utils import get_eyegaze_point_at_depth
from common import ROOT, OUT, CONFIG, read_json, write_json, digest, encoder, finish_encoder


def poses(path, times, RDC):
    df = pd.read_csv(path, usecols=['tracking_timestamp_us','graph_uid','quality_score']+
                     [f'q{x}_world_device' for x in 'xyzw']+[f't{x}_world_device' for x in 'xyz'])
    ts = df.tracking_timestamp_us.to_numpy(np.int64)*1000
    assert (np.diff(ts)>0).all(), 'MPS timestamps must increase'
    n = len(times)
    rotations = np.full((n,3,3),np.nan)
    trans = np.full((n,3),np.nan)
    audit = []
    quats = df[[f'q{x}_world_device' for x in 'xyzw']].to_numpy()
    xyz = df[[f't{x}_world_device' for x in 'xyz']].to_numpy()
    quality = df.quality_score.to_numpy()
    session = df.graph_uid.to_numpy()
    for i,t in enumerate(times):
        r = int(np.searchsorted(ts,t))
        l = r if r < len(ts) and ts[r] == t else r-1
        status = 'valid'
        a = {'frame':i,'device_timestamp_ns':int(t)}
        if l < 0 or r >= len(ts): status='outside_pose_range'
        else:
            a.update(left_gap_ms=float((t-ts[l])/1e6),right_gap_ms=float((ts[r]-t)/1e6),quality=float(min(quality[l],quality[r])),graph_uid=str(session[l]))
            if session[l] != session[r]: status='graph_boundary'
            elif max(t-ts[l],ts[r]-t)>5_000_000: status='gap_above_5ms'
            elif not np.isfinite(quality[[l,r]]).all() or min(quality[l],quality[r])<.5: status='quality_below_0.5'
            elif not np.isfinite(quats[[l,r]]).all(): status='invalid_quaternion'
            else:
                f = 0. if l==r else (t-ts[l])/(ts[r]-ts[l])
                rd = Rotation.from_quat(quats[l]).as_matrix() if l==r else Slerp([0,1],Rotation.from_quat(quats[[l,r]]))(f).as_matrix()
                rotations[i] = rd@RDC
                trans[i] = xyz[l]*(1-f)+xyz[r]*f
        a['status']=status
        audit.append(a)
    return rotations,trans,audit


def gazes(path, times, device_cal, target_cal):
    gaze_data = mps.read_eyegaze(str(path))
    ts = np.array([int(g.tracking_timestamp.total_seconds()*1e9) for g in gaze_data],dtype=np.int64)
    T_camera_cpf = target_cal.get_transform_device_camera().inverse() @ device_cal.get_transform_device_cpf()
    xy = np.full((len(times),2),np.nan)
    xy_depth = xy.copy()
    angles = xy.copy()
    audit=[]
    for i,t in enumerate(times):
        r = int(np.searchsorted(ts,t))
        candidates = [j for j in (r-1,r) if 0<=j<len(ts)]
        if not candidates:
            audit.append({'frame':i,'status':'no_gaze'}); continue
        j = min(candidates,key=lambda k:abs(int(ts[k])-int(t)))
        g = gaze_data[j]
        gap=abs(int(ts[j])-int(t))/1e6
        row={'frame':i,'gap_ms':gap,'status':'valid_direction','spatial_valid':bool(g.spatial_gaze_point_valid)}
        row['gaze_sample_timestamp_ns']=int(ts[j])
        row['signed_gap_ms']=float((int(ts[j])-int(t))/1e6)
        if gap>100 or not np.isfinite([g.yaw,g.pitch]).all() or not g.combined_gaze_valid:
            row['status']='invalid_or_gap'; audit.append(row); continue
        # Depth-free parallel-ray approximation, not an invented fixation depth.
        direction=np.asarray(get_eyegaze_point_at_depth(g.yaw,g.pitch,1.0))
        ray=T_camera_cpf.rotation().to_matrix()@direction
        projected=target_cal.project(ray)
        if projected is not None:
            xy[i]=projected
            angles[i]=[g.yaw,g.pitch]
        else: row['status']='projection_outside'
        if g.spatial_gaze_point_valid:
            p=target_cal.project(T_camera_cpf@g.spatial_gaze_point_in_cpf)
            if p is not None: xy_depth[i]=p
        audit.append(row)
    return xy,xy_depth,angles,audit


def export(spec, alignment_only=False):
    target=OUT/'clips'/spec['id']
    if not alignment_only and (target/'manifest.json').exists() and read_json(target/'manifest.json').get('status')=='complete':
        print('REUSE',spec['id'],flush=True); return
    target.mkdir(parents=True,exist_ok=True)
    vrs=ROOT/'data/Aria/raw_vrs'/f"{spec['recording']}.vrs"
    mpsdir=vrs.parent/f"mps_{spec['recording']}_vrs"
    provider=data_provider.create_vrs_data_provider(str(vrs))
    sid=provider.get_stream_id_from_label('camera-rgb')
    dev=provider.get_device_calibration()
    src=dev.get_camera_calib('camera-rgb')
    w,h=map(int,src.get_image_size())
    dst=calibration.get_linear_camera_calibration(w,h,float(src.get_focal_lengths()[0]),'pinhole',src.get_transform_device_camera())
    fps=float(provider.get_image_configuration(sid).nominal_rate_hz)
    assert fps==10, 'Frozen frame protocol is 10Hz'
    ts=np.array(provider.get_timestamps_ns(sid,TimeDomain.DEVICE_TIME),dtype=np.int64)
    start=int(round(spec['start_s']*fps)); stop=start+int(round(spec['duration_s']*fps))
    assert 0<=start<stop<=len(ts)
    indices=np.arange(start,stop); times=ts[indices]
    assert (np.diff(times)>0).all()
    K=np.eye(3); K[0,0],K[1,1]=dst.get_focal_lengths(); K[:2,2]=dst.get_principal_point()
    extr=dst.get_transform_device_camera().to_matrix()
    rotations,trans,pose_audit=poses(mpsdir/'slam/closed_loop_trajectory.csv',times,extr[:3,:3])
    gaze_xy,gaze_depth_xy,gaze_angles,gaze_audit=gazes(mpsdir/'eye_gaze/general_eye_gaze.csv',times,dev,dst)
    if alignment_only:
        for name in ('reference.npz','alignment_audit.json','manifest.json'):
            old=target/name
            backup=old.with_name(old.stem+'_initial25ms'+old.suffix)
            if old.exists() and not backup.exists():
                import shutil
                shutil.copy2(old,backup)
    np.savez_compressed(target/'reference.npz',times_ns=times,source_indices=indices,K=K,T_device_camera=extr,
                        R_world_camera=rotations,translation_world_device=trans,gaze_xy_direction=gaze_xy,
                        gaze_xy_spatial=gaze_depth_xy,gaze_angles_cpf=gaze_angles)
    write_json(target/'alignment_audit.json',{'pose':pose_audit,'gaze':gaze_audit})
    if alignment_only:
        manifest=read_json(target/'manifest.json')
        manifest.update(valid_gaze_direction_frames=int(np.isfinite(gaze_xy).all(axis=1).sum()),
                        valid_gaze_spatial_frames=int(np.isfinite(gaze_depth_xy).all(axis=1).sum()),
                        alignment_version='MPS5Hz_nearest100ms',alignment_script_sha256=digest(Path(__file__)),
                        alignment_config_sha256=digest(CONFIG))
        write_json(target/'manifest.json',manifest)
        print('REALIGNED',spec['id'],manifest['valid_gaze_direction_frames'],flush=True)
        return
    with (target/'frame_timestamps.csv').open('w',newline='',encoding='utf-8') as f:
        wr=csv.writer(f); wr.writerow(['frame','source_rgb_index','device_timestamp_ns','relative_recording_s'])
        wr.writerows((i,int(idx),int(t),float((t-ts[0])/1e9)) for i,(idx,t) in enumerate(zip(indices,times)))
    proc,log=encoder(target/'rgb.mp4',w,h,fps,18)
    thumbnails=[]; before=vrs.stat(); began=time.monotonic()
    for i,idx in enumerate(indices):
        img,_=provider.get_image_data_by_index(sid,int(idx))
        rgb=calibration.distort_by_calibration(img.to_numpy_array(),dst,src)
        bgr=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
        proc.stdin.write(bgr.tobytes())
        if i%20==0:
            small=cv2.resize(bgr,(512,384))
            cv2.putText(small,f"{spec['id']} f={i}",(10,25),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,255,255),1)
            thumbnails.append(small)
            print(spec['id'],i,'/',len(indices),flush=True)
    finish_encoder(proc,log)
    thumbnails.append(np.zeros_like(thumbnails[0]))
    cv2.imwrite(str(target/'contact_sheet.jpg'),np.vstack([np.hstack(thumbnails[:3]),np.hstack(thumbnails[3:6])]))
    after=vrs.stat()
    assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
    cp=cv2.VideoCapture(str(target/'rgb.mp4'))
    frames=int(cp.get(cv2.CAP_PROP_FRAME_COUNT));cp.release();assert frames==len(times)
    manifest={'status':'complete','spec':spec,'source_vrs':str(vrs),'source_size':before.st_size,'source_mtime_ns':before.st_mtime_ns,
              'config_sha256':digest(CONFIG),'script_sha256':digest(Path(__file__)),'video_sha256':digest(target/'rgb.mp4'),
              'fps':fps,'frames':frames,'width':w,'height':h,'K':K.tolist(),'T_device_camera':extr.tolist(),
              'valid_pose_frames':sum(a['status']=='valid' for a in pose_audit),
              'valid_gaze_direction_frames':int(np.isfinite(gaze_xy).all(axis=1).sum()),
              'valid_gaze_spatial_frames':int(np.isfinite(gaze_depth_xy).all(axis=1).sum()),
              'gaze_model':'general MPS, no personal calibration; direction projection ignores origin baseline; spatial projection uses valid SDK point',
              'pose_reference':'offline closed-loop MPS world-device, <=5ms brackets, quality>=0.5; no real-time claim',
              'seconds':time.monotonic()-began,'projectaria_tools':importlib.metadata.version('projectaria-tools')}
    write_json(target/'manifest.json',manifest)
    print('COMPLETE',spec['id'],manifest['valid_pose_frames'],manifest['valid_gaze_direction_frames'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--clip');parser.add_argument('--alignment-only',action='store_true');args=parser.parse_args()
    config=read_json(CONFIG)
    for spec in config['clips']:
        if not args.clip or spec['id']==args.clip: export(spec,args.alignment_only)
