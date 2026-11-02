"""Pure Aria rotation projection on rectified RGB; KLT is evaluation only.

No depth, learned association, RGB transform fitting, or future trajectory prediction.
All pixel errors are reported in native 2560x1920 coordinates.
"""
import argparse
import csv
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'outputs/tracktrack_reproduction/aria_164557_first60s_v1'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def warp(points, K, R):
    rays = np.c_[points, np.ones(len(points))] @ np.linalg.inv(K).T
    rotated = rays @ R.T
    projected = rotated @ K.T
    valid = np.isfinite(projected).all(axis=1) & (rotated[:, 2] > 1e-8)
    out = np.full((len(points), 2), np.nan)
    out[valid] = projected[valid, :2] / projected[valid, 2:3]
    return out


def self_test(K, RDC):
    points = np.array([[1280., 960.], [200., 200.], [2300., 1500.]])
    R = Rotation.from_rotvec([.02, -.08, .01]).as_matrix()
    identity = np.max(np.abs(warp(points, K, np.eye(3)) - points))
    inverse = np.max(np.abs(warp(warp(points, K, R), K, R.T) - points))
    H = K @ R @ np.linalg.inv(K)
    q = np.c_[points, np.ones(len(points))] @ H.T
    equivalent = np.max(np.abs(warp(points, K, R) - q[:, :2] / q[:, 2:3]))
    # Camera turns right around its y axis: an initially central world ray moves left.
    yaw = .1
    q = warp(np.array([[K[0, 2], K[1, 2]]]), K, Rotation.from_rotvec([0, -yaw, 0]).as_matrix())[0]
    yaw_error = abs(q[0] - (K[0, 2] - K[0, 0] * np.tan(yaw)))
    # Non-identity device extrinsic: derive world-device from chosen world-camera.
    RWC0 = Rotation.from_rotvec([.2, .1, -.3]).as_matrix()
    RWC1 = RWC0 @ R.T
    RWD0, RWD1 = RWC0 @ RDC.T, RWC1 @ RDC.T
    recovered = (RWD1 @ RDC).T @ (RWD0 @ RDC)
    extrinsic = np.max(np.abs(recovered - R))
    values = dict(identity_px=float(identity), roundtrip_px=float(inverse),
                  homography_equivalence_px=float(equivalent), yaw_direction_px=float(yaw_error),
                  extrinsic_rotation_error=float(extrinsic))
    assert max(values.values()) < 1e-8, values
    return values


def align_poses(path, times):
    with path.open(newline='', encoding='utf-8-sig') as f:
        data = list(csv.DictReader(f))
    t = np.array([int(r['tracking_timestamp_us']) * 1000 for r in data], np.int64)
    if np.any(np.diff(t) <= 0):
        raise ValueError('Pose timestamps must increase; split sessions before proceeding')
    qualities = np.array([float(r['quality_score']) for r in data])
    quats = np.array([[float(r[f'q{k}_odometry_device']) for k in 'xyzw'] for r in data])
    translations = np.array([[float(r[f't{k}_odometry_device']) for k in 'xyz'] for r in data])
    sessions = [r['session_uid'] for r in data]
    aligned, audits = [], []
    for frame, query in enumerate(times):
        right = int(np.searchsorted(t, query))
        left = right if right < len(t) and t[right] == query else right - 1
        reason, pose = 'valid', None
        row = dict(frame=frame, device_timestamp_ns=int(query))
        if left < 0 or right >= len(t):
            reason = 'outside_vio_time_range'
        else:
            row.update(left_gap_ms=float((query-t[left])/1e6), right_gap_ms=float((t[right]-query)/1e6),
                       quality_min=float(min(qualities[left], qualities[right])), session=sessions[left])
            if not sessions[left] or sessions[left] != sessions[right]:
                reason = 'session_boundary'
            elif max(query-t[left], t[right]-query) > 5_000_000:
                reason = 'sample_gap_above_5ms'
            elif not np.isfinite(qualities[[left, right]]).all() or min(qualities[left], qualities[right]) < .5:
                reason = 'quality_below_0.5'
            elif not np.isfinite(quats[[left, right]]).all() or np.any(np.linalg.norm(quats[[left, right]], axis=1) < .1):
                reason = 'invalid_quaternion'
            else:
                frac = 0 if left == right else (query-t[left])/(t[right]-t[left])
                rot = Rotation.from_quat(quats[left]).as_matrix() if left == right else Slerp(
                    [0, 1], Rotation.from_quat(quats[[left, right]]))(frac).as_matrix()
                pose = (rot, translations[left]*(1-frac)+translations[right]*frac, sessions[left])
        row['status'] = reason
        aligned.append(pose)
        audits.append(row)
    return aligned, audits


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def background_mask(gray, dets, scale):
    # Exclude black rectification boundary + a margin and all cached person candidates.
    mask = np.uint8(gray > 3)*255
    mask = cv2.erode(mask, np.ones((15, 15), np.uint8))
    mask[:12] = mask[-12:] = 0
    mask[:, :12] = mask[:, -12:] = 0
    h, w = gray.shape
    for d in dets:
        x1, y1, x2, y2 = d[:4]*scale
        dx, dy = max(6, .15*(x2-x1)), max(6, .15*(y2-y1))
        mask[max(0,int(y1-dy)):min(h,int(y2+dy)+1), max(0,int(x1-dx)):min(w,int(x2+dx)+1)] = 0
    return mask


def correspondences(prev, cur, mask0, mask1, scale):
    p = cv2.goodFeaturesToTrack(prev, maxCorners=1000, qualityLevel=.012, minDistance=14,
                                mask=mask0, blockSize=7)
    if p is None:
        return np.empty((0,2)), np.empty((0,2)), 0
    # Deliberately no Aria initialization and no geometric RANSAC selection.
    settings = dict(winSize=(31,31), maxLevel=5,
                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, .01))
    q, st, _ = cv2.calcOpticalFlowPyrLK(prev, cur, p, None, **settings)
    if q is None:
        return np.empty((0,2)), np.empty((0,2)), len(p)
    back, stback, _ = cv2.calcOpticalFlowPyrLK(cur, prev, q, None, **settings)
    a, b = p[:,0,:], q[:,0,:]
    valid = (st.ravel() == 1) & (stback.ravel() == 1) & np.isfinite(b).all(axis=1)
    valid &= np.linalg.norm(back[:,0,:]-a, axis=1) <= 1.0
    h,w = cur.shape
    # Check both endpoints outside foreground; do not select by rotation residual.
    for i in np.flatnonzero(valid):
        x,y = np.rint(b[i]).astype(int)
        if x < 10 or y < 10 or x >= w-10 or y >= h-10 or mask1[y,x] == 0:
            valid[i] = False
            continue
        pa = cv2.getRectSubPix(prev, (15,15), tuple(map(float,a[i]))).astype(float).ravel()
        pb = cv2.getRectSubPix(cur, (15,15), tuple(map(float,b[i]))).astype(float).ravel()
        pa -= pa.mean(); pb -= pb.mean()
        denom = np.linalg.norm(pa)*np.linalg.norm(pb)
        if denom < 1 or np.dot(pa,pb)/denom < .75:
            valid[i] = False
    # OpenCV resize pixel center convention: native=(small+0.5)/scale-0.5.
    return (a[valid]+.5)/scale-.5, (b[valid]+.5)/scale-.5, len(p)


def panel(frame, points, observed, predicted, label, color, stats, valid=True):
    out = cv2.resize(frame, (768,576), interpolation=cv2.INTER_AREA)
    factor = 768/frame.shape[1]
    # Fixed spatial-order subsample, not chosen by residual or by a winning method.
    if len(points):
        order = np.lexsort((points[:,0], points[:,1]))
        inds = order[np.linspace(0,len(order)-1,min(65,len(order)),dtype=int)]
        for i in inds:
            q = tuple(np.rint((observed[i]+.5)*factor-.5).astype(int))
            p = tuple(np.rint((predicted[i]+.5)*factor-.5).astype(int))
            if 0 <= p[0] < 768 and 0 <= p[1] < 576:
                cv2.line(out,p,q,color,1,cv2.LINE_AA)
                cv2.drawMarker(out,p,color,cv2.MARKER_CROSS,7,1)
            cv2.circle(out,q,2,(0,255,0),-1,cv2.LINE_AA)
    out = cv2.copyMakeBorder(out,64,0,0,0,cv2.BORDER_CONSTANT,value=(20,20,20))
    cv2.putText(out,label,(12,23),cv2.FONT_HERSHEY_SIMPLEX,.6,color,1,cv2.LINE_AA)
    cv2.putText(out,stats,(12,48),cv2.FONT_HERSHEY_SIMPLEX,.43,(255,255,255),1,cv2.LINE_AA)
    if not valid:
        cv2.putText(out,'NO VALID ARIA POSE - NOT SCORED',(45,340),cv2.FONT_HERSHEY_SIMPLEX,.7,(0,200,255),2)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--frames', type=int, default=600)
    parser.add_argument('--ffmpeg', default='E:/下载/ffmpeg-2026-06-15-git-44d082edc8-full_build/bin/ffmpeg.exe')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a new output directory; preserving previous runs')
    args.output.mkdir(parents=True)
    cv2.setNumThreads(4)
    manifest_path = BASE/'input/manifest.json'
    metadata = json.loads(manifest_path.read_text(encoding='utf-8'))
    K = np.array(metadata['geometry']['target']['K'])
    TDC = np.array(metadata['geometry']['target']['T_device_camera'])
    RDC = TDC[:3,:3]
    tests = self_test(K,RDC)
    with (BASE/'input/frame_timestamps.csv').open(encoding='utf-8-sig',newline='') as f:
        timestamp_rows = list(csv.DictReader(f))[:args.frames]
    times = np.array([int(r['device_timestamp_ns']) for r in timestamp_rows])
    assert len(times)==args.frames and np.all(np.diff(times)>0)
    assert [int(r['source_rgb_index_zero_based']) for r in timestamp_rows] == list(range(args.frames))
    trajectory = Path('E:/Aria/parsed/unknown_20260803_164557/csv/slam/open_loop_trajectory.csv')
    poses, audits = align_poses(trajectory,times)
    write_csv(args.output/'pose_validity.csv',audits)
    video = BASE/'input/aria_rgb_rectified_first60s.mp4'
    cap = cv2.VideoCapture(str(video))
    assert cap.isOpened()
    fps = cap.get(cv2.CAP_PROP_FPS)
    outvideo = args.output/'rotation_background_comparison.mp4'
    command = [args.ffmpeg,'-hide_banner','-loglevel','error','-n','-f','rawvideo','-pix_fmt','bgr24',
               '-s','1536x640','-r',str(fps),'-i','pipe:0','-an','-c:v','libx264','-preset','fast',
               '-crf','19','-pix_fmt','yuv420p','-threads','4','-movflags','+faststart',str(outvideo)]
    encoder = subprocess.Popen(command,stdin=subprocess.PIPE)
    frame_rows, point_rows, snapshots = [], [], []
    rotations = np.full((args.frames,3,3),np.nan)
    previous_gray=previous_mask=None
    try:
        for index in range(args.frames):
            ok,frame = cap.read()
            if not ok or frame.shape[:2]!=(1920,2560):
                raise ValueError(f'Invalid frame {index}')
            gray=cv2.cvtColor(cv2.resize(frame,(1280,960),interpolation=cv2.INTER_AREA),cv2.COLOR_BGR2GRAY)
            with np.load(BASE/f'tracking/detections/{index+1:06d}.npz') as archive:
                dets=archive['det95']
            mask=background_mask(gray,dets,.5)
            a=b=pred=np.empty((0,2))
            row=dict(frame=index,device_time_ns=int(times[index]),status='first_frame')
            valid=index>0 and poses[index] is not None and poses[index-1] is not None
            if valid and poses[index][2]!=poses[index-1][2]:
                valid=False
            stats=f'frame {index}/{args.frames-1} | device {(times[index]-times[0])/1e9:.2f}s'
            if valid:
                R0,t0,_=poses[index-1]; R1,t1,_=poses[index]
                R=(R1@RDC).T@(R0@RDC)
                rotations[index]=R
                a,b,detected=correspondences(previous_gray,gray,previous_mask,mask,.5)
                pred=warp(a,K,R)
                inverse=warp(a,K,R.T)
                finite=np.isfinite(pred).all(axis=1)&np.isfinite(inverse).all(axis=1)
                a,b,pred,inverse=a[finite],b[finite],pred[finite],inverse[finite]
                theta=float(np.rad2deg(np.linalg.norm(Rotation.from_matrix(R).as_rotvec())))
                # Includes device-camera lever arm; diagnostic only, never used for pixel warp.
                translation=float(np.linalg.norm((t1+R1@TDC[:3,3])-(t0+R0@TDC[:3,3])))
                row.update(status='valid' if len(a)>=20 else 'insufficient_background_candidates',
                           detected_features=detected,accepted_points=len(a),angle_deg=theta,
                           camera_translation_m=translation,dt_ms=float((times[index]-times[index-1])/1e6))
                if len(a):
                    errors=[np.linalg.norm(x-b,axis=1) for x in (a,pred,inverse)]
                    for name,err in zip(['none','rotation','inverse'],errors):
                        row[name+'_median_px']=float(np.median(err))
                        row[name+'_p90_px']=float(np.quantile(err,.9))
                    for j in range(len(a)):
                        point_rows.append(dict(frame=index,point=j,x0=a[j,0],y0=a[j,1],x_observed=b[j,0],y_observed=b[j,1],
                                               x_rotation=pred[j,0],y_rotation=pred[j,1],error_none_px=errors[0][j],
                                               error_rotation_px=errors[1][j],error_inverse_px=errors[2][j]))
                    stats+=f' | N={len(a)} | rot={theta:.2f}deg'
                leftstats=stats+f' | median={row.get("none_median_px",float("nan")):.1f}px'
                rightstats=stats+f' | median={row.get("rotation_median_px",float("nan")):.1f}px'
            else:
                if index:
                    row['status']='invalid_pose_pair'
                leftstats=rightstats=stats+' | '+row['status']
            left=panel(frame,a,b,a,'NO COMPENSATION: red cross / green observed', (0,80,255),leftstats,valid)
            right=panel(frame,a,b,pred,'ARIA ROTATION: blue cross / green observed',(255,190,30),rightstats,valid)
            preview=np.hstack([left,right])
            encoder.stdin.write(preview.tobytes())
            if index in [23,50,76,150,300,450,599]:
                cv2.imwrite(str(args.output/f'frame_{index:06d}.jpg'),preview)
                snapshots.append(preview)
            frame_rows.append(row)
            previous_gray,previous_mask=gray,mask
            if (index+1)%100==0:
                print(f'Processed {index+1}/{args.frames}',flush=True)
    finally:
        cap.release()
        encoder.stdin.close()
        status=encoder.wait()
    if status:
        raise RuntimeError(f'FFmpeg failed: {status}')
    write_csv(args.output/'frame_residuals.csv',frame_rows)
    write_csv(args.output/'point_residuals.csv',point_rows)
    np.savez_compressed(args.output/'relative_rotations.npz',R_current_from_previous=rotations,K_rect=K)
    usable=[r for r in frame_rows if r['status']=='valid']
    summary={'evaluated_pairs':len(usable),'min_points_per_pair':20,
             'frame_status_counts':dict(Counter(r['status'] for r in frame_rows)),
             'pose_status_counts':dict(Counter(r['status'] for r in audits))}
    for name in ['none','rotation','inverse']:
        summary[name]={k:float(np.median([r[name+'_'+k+'_px'] for r in usable])) for k in ['median','p90']}
    summary['improved_pair_fraction']=float(np.mean([r['rotation_median_px']<r['none_median_px'] for r in usable]))
    summary['paired_median_improvement_px']=float(np.median([r['none_median_px']-r['rotation_median_px'] for r in usable]))
    # Rotation groups are descriptive within-recording groups, not independent tests.
    cuts=np.quantile([r['angle_deg'] for r in usable],[1/3,2/3])
    summary['rotation_group_cutoffs_deg']=cuts.tolist()
    summary['rotation_groups']={}
    for label,lo,hi in [('low',-1,cuts[0]),('mid',cuts[0],cuts[1]),('high',cuts[1],float('inf'))]:
        group=[r for r in usable if lo<r['angle_deg']<=hi]
        summary['rotation_groups'][label]={'pairs':len(group),**{name:float(np.median([r[name+'_median_px'] for r in group])) for name in ['none','rotation','inverse']}}
    check=cv2.VideoCapture(str(outvideo)); decoded=0
    while check.read()[0]:
        decoded+=1
    rate=check.get(cv2.CAP_PROP_FPS); check.release()
    assert decoded==args.frames and abs(rate-fps)<1e-6
    provenance={'video':str(video),'video_sha256':sha(video),'rectification_manifest':str(manifest_path),
                'rectification_manifest_sha256':sha(manifest_path),'time_map_sha256':sha(BASE/'input/frame_timestamps.csv'),
                'trajectory':str(trajectory),'trajectory_sha256':sha(trajectory),'script_sha256':sha(__file__),
                'K_rect':K.tolist(),'T_device_camera':TDC.tolist(),'self_tests':tests,
                'video_verified_frames':decoded,'fps':fps,'summary':summary,
                'constraints':['Pure rotation only; translation recorded for diagnosis, not applied',
                               'Same unseeded KLT correspondences for all three conditions; no transform fitting',
                               'Person masks use cached YOLOX det95, padded 15%; not ground-truth masks',
                               'KLT forward-back <=1px at half resolution; patch NCC >=0.75',
                               'Foreground leakage and KLT failure/selection bias remain; not static ground truth',
                               'Offline pose interpolation within 5ms; not an online latency claim',
                               'No depth; no RGB GMC; no identity tracking; original data unchanged']}
    (args.output/'manifest.json').write_text(json.dumps(provenance,indent=2,ensure_ascii=False),encoding='utf-8')
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    main()
