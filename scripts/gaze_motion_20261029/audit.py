"""Official-loader gaze audit plus same-time MPS join; no behavioral inference.
Run in WSL /opt/aria-vrs/bin/python, with explicit shared --data-root and --out.
"""
import argparse, csv, json, hashlib, importlib.metadata
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from projectaria_tools.core import data_provider, calibration, mps
from projectaria_tools.core.mps.utils import get_eyegaze_point_at_depth

def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data-root',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True); records=[];cache={};hashes={}
    raw=a.data_root/'data/Aria/raw_vrs'
    for vrs in sorted(raw.glob('*.vrs')):
        rec=vrs.stem; folder=raw/f'mps_{rec}_vrs';gp=folder/'eye_gaze/general_eye_gaze.csv'
        if not gp.exists():continue
        gs=mps.read_eyegaze(str(gp)); provider=data_provider.create_vrs_data_provider(str(vrs))
        dev=provider.get_device_calibration();rgb=dev.get_camera_calib('camera-rgb'); w,h=map(int,rgb.get_image_size())
        pin=calibration.get_linear_camera_calibration(w,h,float(rgb.get_focal_lengths()[0]),'pinhole',rgb.get_transform_device_camera())
        Tdc=rgb.get_transform_device_camera().to_matrix();Tdf=dev.get_transform_device_cpf().to_matrix()
        Tcf=np.linalg.inv(Tdc)@Tdf
        K=np.eye(3); K[0,0],K[1,1]=pin.get_focal_lengths();K[:2,2]=pin.get_principal_point()
        ts=np.array([round(g.tracking_timestamp.total_seconds()*1e6)*1000 for g in gs],dtype=np.int64)
        valid=np.array([g.combined_gaze_valid and np.isfinite([g.yaw,g.pitch]).all() for g in gs])
        spatial=np.array([g.spatial_gaze_point_valid for g in gs]);depth=np.array([g.depth for g in gs])
        directions=np.array([get_eyegaze_point_at_depth(g.yaw,g.pitch,1.) for g in gs],float)
        directions/=np.linalg.norm(directions,axis=1,keepdims=True)
        rays=directions@Tcf[:3,:3].T; q=rays@K.T
        pixels=q[:,:2]/q[:,2:]; inside=valid&(rays[:,2]>0)&(pixels[:,0]>=0)&(pixels[:,0]<w)&(pixels[:,1]>=0)&(pixels[:,1]<h)
        posepath=folder/'slam/closed_loop_trajectory.csv';n=len(gs)
        joined=np.zeros(n,bool);age=np.full(n,np.inf);world=np.full((n,3),np.nan);pos=np.full((n,3),np.nan)
        # Missing pose remains missing. No forward/back filling across loss or outside support.
        item={'recording':rec,'rows':n,'valid_direction':int(valid.sum()),'in_linear_rgb':int(inside.sum()),
              'spatial_valid':int(spatial.sum()),'positive_depth':int((np.isfinite(depth)&(depth>0)).sum()),
              'depth_valid_quantiles_m':np.quantile(depth[spatial&np.isfinite(depth)&(depth>0)],[.05,.5,.95]).tolist() if (spatial&np.isfinite(depth)&(depth>0)).any() else [],
              'gaze_median_interval_ms':float(np.median(np.diff(ts))/1e6)}
        if posepath.exists():
            hashes[str(posepath.relative_to(a.data_root))]=hashlib.sha256(posepath.read_bytes()).hexdigest()
            fields=['tracking_timestamp_us','tx_world_device','ty_world_device','tz_world_device','qx_world_device','qy_world_device','qz_world_device','qw_world_device','quality_score']
            with posepath.open() as f:
                reader=csv.reader(f);head=next(reader);idx=[head.index(k) for k in fields]
                vals=np.array([[float(row[j]) for j in idx] for row in reader],dtype=np.float64)
            pts=np.rint(vals[:,0]*1000).astype(np.int64);assert (np.diff(pts)>0).all()
            j=np.searchsorted(pts,ts,side='right')-1;ok=j>=0; jj=np.clip(j,0,len(pts)-1)
            age[ok]=(ts[ok]-pts[jj[ok]])/1e6
            joined=ok&(age<=20)&(vals[jj,8]>0)&valid
            R=Rotation.from_quat(vals[jj,4:8]).as_matrix()
            cpfworld=R@Tdf[:3,:3];world[joined]=np.einsum('nij,nj->ni',cpfworld[joined],directions[joined])
            pos[joined]=vals[jj[joined],1:4]+np.einsum('nij,j->ni',R[joined],Tdc[:3,3])
            item.update(pose_rows=len(pts),pose_span_s=float((pts[-1]-pts[0])/1e9),
                valid_join_20ms=int(joined.sum()),pose_age_join_p95_ms=float(np.percentile(age[joined],95)) if joined.any() else None,
                pose_quality_counts={str(k):int(v) for k,v in zip(*np.unique(vals[:,8],return_counts=True))})
        else:item.update(pose_rows=0,pose_span_s=0.,valid_join_20ms=0)
        # Genuine 16 native samples (3s), sampled nonoverlapping; candidates only, no G1/behavior labels.
        count=0
        for start in range(0,n-15,16):
            sl=slice(start,start+16)
            if joined[sl].all() and (np.diff(ts[sl])<=300_000_000).all():count+=1
        item['nonoverlap_3s_joint_blocks']=count
        np.savez_compressed(a.out/f'{rec}.npz',timestamp_ns=ts,direction_cpf=directions,direction_world=world,
            rgb_center_world=pos,valid=valid,pose_valid=joined,pose_age_ms=age,pixel_direction=pixels,in_image=inside,
            spatial_valid=spatial,derived_gaze_depth=depth,T_device_rgb=Tdc,T_device_cpf=Tdf,K=K)
        hashes[str(gp.relative_to(a.data_root))]=hashlib.sha256(gp.read_bytes()).hexdigest()
        cache[rec]=(ts,valid,inside,joined); records.append(item);print(json.dumps(item),flush=True)
    cohortrows={}
    for name,path in [('short','outputs/g1_expanded_forecast_20261029/expanded_samples.json'),('long','outputs/horizon_15_15_g1_20261029/samples.json')]:
        rows=[]
        for m in json.loads((a.data_root/path).read_text(encoding='utf-8-sig')):
            ts,valid,inside,joined=cache[m['recording']];lo,hi=int(m['timestamps_ns'][0]),int(m['timestamps_ns'][15]); ix=(ts>=lo)&(ts<=hi)
            rows.append({'key':m['key'],'recording':m['recording'],'native_rows':int(ix.sum()),'valid_direction':int((ix&valid).sum()),'in_image':int((ix&inside).sum()),'pose_joined':int((ix&joined).sum())})
        cohortrows[name]=rows
    result={'version':importlib.metadata.version('projectaria-tools'),'records':records,'cohorts':cohortrows,
        'total':{k:sum(r.get(k,0) for r in records) for k in ['rows','valid_direction','in_linear_rgb','spatial_valid','positive_depth','valid_join_20ms','nonoverlap_3s_joint_blocks','pose_span_s']},
        'semantics':'Official flags are validity, not accuracy. Pixel direction is parallel-ray approximation, not exact fixation. World gaze uses same-time last-before closed-loop offline pose, not proof of online MPS. No gaze events, identity attention labels or behavior association inferred.'}
    dump(a.out/'audit.json',result);dump(a.out/'source_hashes.json',hashes);print('TOTAL',json.dumps(result['total']),flush=True)
if __name__=='__main__':main()
