"""Bounded, pretrained official Aria stereo validation; never edits source assets."""
import os, sys, json, time, hashlib, gzip, argparse
from pathlib import Path
import numpy as np
import cv2
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/aria_stereo_validation_20261008'
VENDOR=ROOT/'external/aria_stereo_20261008'
sys.path.insert(0,str(VENDOR))
import export_depth_from_stereo as official
official.load_runtime_dependencies()
from projectaria_tools.core import calibration

def dump(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False))

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()

def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    clips=['03','05','07']; entries=[]; protected={}
    for cid in clips:
        inp=ROOT/f'outputs/sam2_g1_v6_v8_20261001/{cid}/input'
        meta=json.loads((inp/'metadata.json').read_text())
        ref=np.load(inp/'reference.npz')
        for p in [inp/'metadata.json',inp/'reference.npz',inp/'rgb.mp4',ROOT/f'outputs/sam2_near_people_20261002/{cid}/final/masks_rle.jsonl.gz']:
            protected[str(p.relative_to(ROOT))]=sha(p)
        for frame in range(21):
            entries.append(dict(clip=cid,frame=frame,recording=meta['source']['id'],rgb_timestamp_ns=int(ref['times_ns'][frame])))
    protocol=dict(clips=clips,frames_per_clip=21,entries=entries,protected=protected,
        selection='Existing clips 03/05/07, first 21 RGB timestamps; fixed before depth inference; two recordings.',
        model='official FoundationStereo vitl 23-51-11, pretrained epoch40; 32 iterations, no finetuning',
        image_size=[512,512],focal_scale=1.25,lr_threshold_px=1.0,
        mask_erosion_px=7,minimum_person_points=30,
        limits='No measured distance GT. LR agreement and temporal variation are consistency diagnostics, not accuracy. Existing masks are offline SAM references.',
        rgb_mapping='Stored RGB intrinsics/extrinsics, interpolated MPS pose at both timestamps, 3D transform and z-buffer; no resized depth approximation.')
    pp=OUT/'protocol.json'
    if pp.exists():assert json.loads(pp.read_text())==protocol
    else:dump(pp,protocol)
    for cid in clips:
        rows=[e for e in entries if e['clip']==cid];rec=rows[0]['recording']
        vrs=ROOT/f'data/Aria/raw_vrs/unknown_20260803_{rec}.vrs'
        provider=official.data_provider.create_vrs_data_provider(str(vrs))
        paths=official.MpsDataPathsProvider(str(ROOT/f'data/Aria/raw_vrs/mps_unknown_20260803_{rec}_vrs')).get_data_paths()
        mps=official.MpsDataProvider(paths)
        ls=provider.get_stream_id_from_label('slam-front-left');rs=provider.get_stream_id_from_label('slam-front-right')
        ref=np.load(ROOT/f'outputs/sam2_g1_v6_v8_20261001/{cid}/input/reference.npz')
        cap=cv2.VideoCapture(str(ROOT/f'outputs/sam2_g1_v6_v8_20261001/{cid}/input/rgb.mp4'))
        for e in rows:
            folder=OUT/cid/f"{e['frame']:03d}";folder.mkdir(parents=True,exist_ok=True)
            ok,rgb=cap.read();assert ok
            ts=e['rgb_timestamp_ns']
            li,lr=provider.get_image_data_by_time_ns(ls,ts,official.TimeDomain.DEVICE_TIME,official.TimeQueryOptions.CLOSEST)
            lt=int(lr.capture_timestamp_ns)
            ri,rr=provider.get_image_data_by_time_ns(rs,lt,official.TimeDomain.DEVICE_TIME,official.TimeQueryOptions.CLOSEST)
            rt=int(rr.capture_timestamp_ns);assert abs(lt-rt)<=1000000
            online=mps.get_online_calibration(lt,official.TimeQueryOptions.CLOSEST)
            lc=online.get_camera_calib('slam-front-left');rc=online.get_camera_calib('slam-front-right')
            tl=lc.get_transform_device_camera().inverse();tr=rc.get_transform_device_camera().inverse()
            linear=official.fisheye_to_linear_calib(lc,focal_scale=1.25,output_width=512,output_height=512,use_original_pp=True)
            rl,rrr=official.create_scanline_rectified_cameras(tl,tr)
            left,right=official.rectify_stereo_pair(li.to_numpy_array(),ri.to_numpy_array(),lc,rc,linear,linear,rl,rrr,official.InterpolationMethod.BILINEAR)
            supportl,supportr=official.rectify_stereo_pair(np.full_like(li.to_numpy_array(),255),np.full_like(ri.to_numpy_array(),255),lc,rc,linear,linear,rl,rrr,official.InterpolationMethod.BILINEAR)
            baseline=float(np.linalg.norm((tl@tr.inverse()).translation()))
            world_cv=official.compute_T_world_rectCam(mps.get_interpolated_closed_loop_pose(lt).transform_world_device,tl,rl).to_matrix()
            world_rgb=mps.get_interpolated_closed_loop_pose(ts).transform_world_device.to_matrix()@ref['T_device_camera']
            rgb_cv=np.linalg.inv(world_rgb)@world_cv
            params=linear.get_projection_params()
            np.savez_compressed(folder/'pair.npz',left=left,right=right,support_left=supportl>=254,support_right=supportr>=254,
                intrinsics=params,baseline_m=baseline,T_rgb_cv=rgb_cv,T_world_cv=world_cv,K_rgb=ref['K'])
            cv2.imwrite(str(folder/'rgb.jpg'),rgb,[cv2.IMWRITE_JPEG_QUALITY,95])
            cv2.imwrite(str(folder/'raw_left.png'),li.to_numpy_array());cv2.imwrite(str(folder/'raw_right.png'),ri.to_numpy_array())
            cv2.imwrite(str(folder/'rectified_left.png'),left);cv2.imwrite(str(folder/'rectified_right.png'),right)
            # Independent feature correspondences: descriptor ratio + mutual agreement, no y-error filtering.
            sift=cv2.SIFT_create(nfeatures=1500)
            kp1,d1=sift.detectAndCompute(left,None);kp2,d2=sift.detectAndCompute(right,None)
            dy=[]
            if d1 is not None and d2 is not None:
                bm=cv2.BFMatcher();forward=bm.knnMatch(d1,d2,k=2);back=bm.match(d2,d1)
                matches=[a for a,b in forward if a.distance<.7*b.distance and back[a.trainIdx].trainIdx==a.queryIdx]
                dy=[abs(kp1[a.queryIdx].pt[1]-kp2[a.trainIdx].pt[1]) for a in matches]
            dump(folder/'metadata.json',dict(**e,left_timestamp_ns=lt,right_timestamp_ns=rt,
                stereo_skew_ms=abs(lt-rt)/1e6,rgb_skew_ms=abs(lt-ts)/1e6,baseline_m=baseline,
                focal_px=float(params[0]),epipolar_matches=len(dy),epipolar_abs_y_errors_px=dy))
            print('PREPARED',cid,e['frame'],flush=True)
        cap.release()

def infer(limit=0):
    import torch
    model,cfg=official.load_foundation_stereo(str(ROOT/'models/foundation_stereo_20261008/23-51-11/model_best_bp2.pth'))
    protocol=json.loads((OUT/'protocol.json').read_text());count=0
    for e in protocol['entries']:
        folder=OUT/e['clip']/f"{e['frame']:03d}"
        if (folder/'depth.npz').exists():continue
        pair=np.load(folder/'pair.npz');torch.cuda.synchronize();start=time.perf_counter()
        disp,lr=official.run_lr_consistency(model,pair['left'],pair['right'],cfg,threshold=1.0)
        torch.cuda.synchronize();elapsed=time.perf_counter()-start
        depth=official.disparity_to_depth(disp,float(pair['baseline_m']),float(pair['intrinsics'][0]))
        y,x=np.indices(disp.shape);xr=x-disp;xi=np.clip(np.nan_to_num(xr),0,511).astype(int)
        support=pair['support_left']&pair['support_right'][y,xi]&(xr>=0)&(xr<=511)
        valid=support&lr&np.isfinite(disp)&(disp>0)&np.isfinite(depth)&(depth>0)
        np.savez_compressed(folder/'depth.npz',disparity_px=disp,depth_m=depth,lr_consistent=lr,support=support,valid=valid)
        dump(folder/'inference.json',dict(seconds=elapsed,valid_fraction=float(valid.mean()),
            consistent_supported_fraction=float((valid&support).sum()/max(1,support.sum())),
            cuda_peak_allocated_bytes=int(torch.cuda.max_memory_allocated()),cuda_peak_reserved_bytes=int(torch.cuda.max_memory_reserved())))
        print('DEPTH',e['clip'],e['frame'],round(elapsed,3),'s valid',round(valid.mean(),3),flush=True)
        count+=1
        if limit and count>=limit:break

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','infer']);ap.add_argument('--limit',type=int,default=0);a=ap.parse_args()
    if a.mode=='prepare':prepare()
    else:infer(a.limit)
