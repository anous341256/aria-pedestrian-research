"""Frozen, small rotation-only forecasting feasibility diagnostic. No training."""
from pathlib import Path
import csv, hashlib, json, sys
from collections import Counter
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation, Slerp
from projectaria_tools.core import data_provider

ROOT=Path(__file__).resolve().parents[2]
SRC=ROOT/'outputs/sam2_fullvideos_20260927'
OUT=ROOT/'outputs/factor_feasibility_20260928'
METHODS=['stationary','raw_cv','comp_cv_predcam','comp_cv_refcam']
PENALTY=10*np.hypot(2560,1920)

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def serialize(a):
    a=np.asarray(a)
    return [[float(v) if np.isfinite(v) else None for v in row] for row in a]
def project(p,H):
    p=np.asarray(p);z=np.c_[p,np.ones(len(p))]@H.T
    ok=np.isfinite(z).all(axis=1)&(z[:,2]>1e-8)
    q=np.full((len(p),2),np.nan);q[ok]=z[ok,:2]/z[ok,2:]
    return q,ok
def slope(p,t):
    tt=t-t.mean();return tt@(p-p.mean(axis=0))/(tt@tt)

def predict_history(p,R,K,seconds):
    """ONLY 6 past points and orientations. Camera history is offline privileged data."""
    assert p.shape==(6,2) and R.shape==(6,3,3) and seconds.shape==(6,)
    ti=seconds-seconds[-1];tau=np.arange(1,16)/10
    raw=p[-1]+tau[:,None]*slope(p,ti)
    ki=np.linalg.inv(K);cur=R[-1]
    q=np.array([project([x],K@cur.T@r@ki)[0][0] for x,r in zip(p,R)])
    ped=p[-1]+tau[:,None]*slope(q,ti)
    # C_j = R_current^T R_j maps camera-j rays to current orientation.
    logs=Rotation.from_matrix(np.array([cur.T@r for r in R])).as_rotvec()
    omega=slope(logs,ti)
    rel=Rotation.from_rotvec(tau[:,None]*omega).as_matrix()
    pred=np.array([project([x],K@r.T@ki)[0][0] for x,r in zip(ped,rel)])
    return {'stationary':np.repeat(p[-1:],15,axis=0),'raw_cv':raw,
            'comp_cv_predcam':pred},ped,q,omega

def recompose_reference(ped,current,future,K):
    ki=np.linalg.inv(K)
    return np.array([project([p],K@r.T@current@ki)[0][0] for p,r in zip(ped,future)])

def align(recording):
    meta=read(SRC/recording/'manifest.json')
    rows=list(csv.DictReader((SRC/recording/'frame_timestamps.csv').open()))
    times=np.array([int(r['device_timestamp_ns']) for r in rows],np.int64)
    vrs=ROOT/'data/Aria/raw_vrs'/f'unknown_20260803_{recording}.vrs'
    vrsstat=vrs.stat()
    dp=data_provider.create_vrs_data_provider(str(vrs))
    cam=dp.get_device_calibration().get_camera_calib('camera-rgb')
    extr=cam.get_transform_device_camera().to_matrix()
    f0,f1,cx,cy=meta['target_calibration'];K=np.array([[f0,0,cx],[0,f1,cy],[0,0,1.]])
    assert tuple(cam.get_image_size())==(meta['width'],meta['height'])
    assert abs(cam.get_focal_lengths()[0]-f0)<1e-7
    posefile=vrs.parent/f'mps_unknown_20260803_{recording}_vrs/slam/closed_loop_trajectory.csv'
    df=pd.read_csv(posefile,usecols=['tracking_timestamp_us','graph_uid','quality_score']+[f'q{x}_world_device' for x in 'xyzw'])
    ts=df.tracking_timestamp_us.to_numpy(np.int64)*1000;assert (np.diff(ts)>0).all()
    quat=df[[f'q{x}_world_device' for x in 'xyzw']].to_numpy();quality=df.quality_score.to_numpy();graphs=df.graph_uid.to_numpy()
    R=np.full((len(times),3,3),np.nan);audit=[]
    for i,t in enumerate(times):
        rr=int(np.searchsorted(ts,t));ll=rr if rr<len(ts) and ts[rr]==t else rr-1
        a={'frame':i,'status':'valid'}
        if ll<0 or rr>=len(ts):a['status']='outside_pose_range'
        else:
            a.update(left_gap_ms=float((t-ts[ll])/1e6),right_gap_ms=float((ts[rr]-t)/1e6),quality=float(min(quality[ll],quality[rr])),graph=str(graphs[ll]))
            if graphs[ll]!=graphs[rr]:a['status']='graph_boundary'
            elif max(t-ts[ll],ts[rr]-t)>5e6:a['status']='gap'
            elif not np.isfinite(quality[[ll,rr]]).all() or min(quality[ll],quality[rr])<.5:a['status']='quality'
            elif not np.isfinite(quat[[ll,rr]]).all() or np.min(np.linalg.norm(quat[[ll,rr]],axis=1))<.1:a['status']='quaternion'
            else:
                frac=0 if ll==rr else (t-ts[ll])/(ts[rr]-ts[ll])
                rd=Rotation.from_quat(quat[ll]).as_matrix() if ll==rr else Slerp([0,1],Rotation.from_quat(quat[[ll,rr]]))(frac).as_matrix()
                R[i]=rd@extr[:3,:3]
        audit.append(a)
    np.savez_compressed(OUT/f'{recording}_reference.npz',R=R,K=K,times_ns=times,T_device_camera=extr)
    write(OUT/f'{recording}_alignment.json',audit)
    info={'recording':recording,'pose_sha256':sha(posefile),'annotations_sha256':sha(SRC/recording/'sam2_raw_v1/tracks.json'),
          'manifest_sha256':sha(SRC/recording/'manifest.json'),'timestamps_sha256':sha(SRC/recording/'frame_timestamps.csv'),
          'source_vrs_size':vrsstat.st_size,'source_vrs_mtime_ns':vrsstat.st_mtime_ns,'K':K.tolist(),'T_device_camera':extr.tolist(),
          'all_frames_pose_status':dict(Counter(a['status'] for a in audit))}
    return R,K,times,audit,info

def metrics(pred,target):
    err=np.linalg.norm(pred-target,axis=1);invalid=~np.isfinite(err);err[invalid]=PENALTY
    d={'invalid_steps':int(invalid.sum()),'errors_px':err.tolist()}
    for n in (5,10,15):d[f'ADE_{n/10:.1f}s']=float(err[:n].mean());d[f'FDE_{n/10:.1f}s']=float(err[n-1])
    return d

def aggregate(rows):
    if not rows:return {'windows':0}
    out={'windows':len(rows),'methods':{}}
    keys=[f'{metric}_{n/10:.1f}s' for n in (5,10,15) for metric in ('ADE','FDE')]
    for m in METHODS:
        d={k:float(np.mean([r['metrics'][m][k] for r in rows])) for k in keys}
        d['median_ADE_1.5s']=float(np.median([r['metrics'][m]['ADE_1.5s'] for r in rows]))
        d['invalid_steps']=sum(r['metrics'][m]['invalid_steps'] for r in rows)
        if m not in ('stationary','raw_cv'):
            delta=np.array([r['metrics'][m]['ADE_1.5s']-r['metrics']['raw_cv']['ADE_1.5s'] for r in rows])
            d['mean_paired_ADE_delta_vs_raw']=float(delta.mean());d['fraction_windows_better_than_raw']=float((delta<0).mean())
        out['methods'][m]=d
    return out

def verification():
    K=np.array([[1100.,0,1279.5],[0,1100.,959.5],[0,0,1.]])
    ts=np.arange(-5,16)/10;omega=np.array([.02,.15,-.01])
    R=Rotation.from_rotvec(ts[:,None]*omega).as_matrix()
    fixed=np.c_[1300+70*ts,900-15*ts]
    raw=np.array([project([p],K@r.T@np.linalg.inv(K))[0][0] for p,r in zip(fixed,R)])
    pred,ped,_,_=predict_history(raw[:6],R[:6],K,ts[:6])
    ref=recompose_reference(ped,R[5],R[6:],K)
    out={'synthetic_compensated_prediction_px':float(np.max(abs(pred['comp_cv_predcam']-raw[6:]))),
         'synthetic_reference_projection_px':float(np.max(abs(ref-raw[6:])))}
    identity=np.repeat(np.eye(3)[None],6,axis=0)
    a,_,_,_=predict_history(raw[:6],identity,K,ts[:6]);out['identity_equals_raw_px']=float(np.max(abs(a['raw_cv']-a['comp_cv_predcam'])))
    rng=np.random.default_rng(5);r=Rotation.from_rotvec([.02,-.08,.01]).as_matrix();p=np.array([[300,400],[1279.5,959.5],[2200,1400.]])
    q,_=project(p,K@r@np.linalg.inv(K));back,_=project(q,K@r.T@np.linalg.inv(K));out['rotation_roundtrip_px']=float(np.max(abs(back-p)))
    shifted,_=project([[1279.5,959.5]],K@Rotation.from_rotvec([0,-.1,0]).as_matrix()@np.linalg.inv(K))
    out['positive_camera_yaw_moves_stationary_center_left']=bool(shifted[0,0]<1279.5)
    p2=raw.copy();r2=R.copy();p2[6:]=rng.normal(size=(15,2))*1e5;r2[6:]=Rotation.random(15,random_state=rng).as_matrix()
    b,_,_,_=predict_history(p2[:6],r2[:6],K,ts[:6]);out['future_suffix_mutation_px']=max(float(np.max(abs(pred[m]-b[m]))) for m in pred)
    ref2=recompose_reference(ped,R[5],r2[6:],K)
    out['future_reference_diagnostic_changes']=bool(not np.allclose(ref,ref2,equal_nan=True))
    assert all(out[k]<1e-7 for k in out if k.endswith('_px')),out
    assert out['positive_camera_yaw_moves_stationary_center_left'] and out['future_reference_diagnostic_changes']
    return out

def main():
    OUT.mkdir(exist_ok=True)
    assert not (OUT/'summary.json').exists(),'Preserve completed results; use a fresh output directory.'
    protocol=read(Path(__file__).with_name('protocol.json'))
    write(OUT/'protocol.json',protocol)
    specs=read(ROOT/'outputs/sam2_forecast_20260928/candidates.json');write(OUT/'candidates.json',specs)
    write(OUT/'verification.json',verification())
    cache={};manifest=[]
    for sid in sorted({s['recording'] for s in specs}):
        print('Align',sid,flush=True);*cache[sid],info=align(sid);manifest.append(info)
    rows=[];excluded=[];all_examples=[]
    for s in specs:
        R,K,ns,audit=cache[s['recording']]
        data=read(SRC/s['recording']/'sam2_raw_v1/tracks.json')
        track=next(t for t in data['tracks'] if t['id']==s['track'])
        boxes={r['frame']:r['bbox_xyxy'] for r in track['frames']}
        for cutoff in range(s['start']+5,s['end']-15+1,5):
            fs=np.arange(cutoff-5,cutoff+16);rr=R[fs]
            status='valid'
            if not np.isfinite(rr).all():status='invalid_reference_pose'
            elif len(set(audit[f]['graph'] for f in fs))!=1:status='reference_graph_boundary'
            if status!='valid':excluded.append({'key':s['key'],'frame':cutoff,'reason':status});continue
            b=np.array([boxes[int(f)] for f in fs],float);points=(b[:,:2]+b[:,2:])/2
            seconds=(ns[fs[:6]]-ns[cutoff])/1e9
            pred,ped,hist,omega=predict_history(points[:6],rr[:6],K,seconds)
            pred['comp_cv_refcam']=recompose_reference(ped,rr[5],rr[6:],K)
            rotation_deg=float(np.max(np.linalg.norm(Rotation.from_matrix(np.array([rr[5].T@r for r in rr[6:]])).as_rotvec(),axis=1))*180/np.pi)
            row={'key':s['key'],'recording':s['recording'],'track':s['track'],'frame':cutoff,
                 'history_frames':fs[:6].tolist(),'future_frames':fs[6:].tolist(),'history_xy':points[:6].tolist(),
                 'history_normalized_xy':serialize(hist),'history_xyxy':b[:6].tolist(),'target_xy':points[6:].tolist(),
                 'pedestrian_forecast_normalized_xy':serialize(ped),'predictions':{m:serialize(p) for m,p in pred.items()},
                 'metrics':{m:metrics(p,points[6:]) for m,p in pred.items()},'camera_omega_rad_s':omega.tolist(),
                 'max_future_rotation_deg':rotation_deg,'edge':bool(((b[:,0]<=5)|(b[:,1]<=5)|(b[:,2]>=2555)|(b[:,3]>=1915)).any()),
                 'nonoverlap':bool((cutoff-(s['start']+5))%20==0),
                 'actual_horizon_s':float((ns[cutoff+15]-ns[cutoff])/1e9)}
            rows.append(row)
        mine=[r for r in rows if r['key']==s['key']]
        if mine:all_examples.append(min(mine,key=lambda r:abs(r['frame']-(s['start']+50))))
        print(s['key'],aggregate(mine),flush=True)
    summary={'overall':aggregate(rows),'per_track':{s['key']:aggregate([r for r in rows if r['key']==s['key']]) for s in specs},
             'per_recording':{sid:aggregate([r for r in rows if r['recording']==sid]) for sid in cache},
             'nonoverlap':aggregate([r for r in rows if r['nonoverlap']]),'no_edge':aggregate([r for r in rows if not r['edge']]),
             'all_methods_projectable':aggregate([r for r in rows if all(r['metrics'][m]['invalid_steps']==0 for m in METHODS)]),
             'rotation_strata':{label:aggregate([r for r in rows if lo<=r['max_future_rotation_deg']<hi]) for label,lo,hi in [('lt5',0,5),('5to15',5,15),('ge15',15,1e9)]},
             'excluded_windows':excluded,'protocol_sha256':sha(OUT/'protocol.json'),'source_manifest':manifest,
             'horizon_seconds_min_max':[min(r['actual_horizon_s'] for r in rows),max(r['actual_horizon_s'] for r in rows)]}
    for name,group in [('equal_track','per_track'),('equal_recording','per_recording')]:
        summary[name]={m:{metric:float(np.mean([v['methods'][m][metric] for v in summary[group].values() if v['windows']])) for metric in ['ADE_1.5s','FDE_1.5s']} for m in METHODS}
    write(OUT/'predictions.json',rows);write(OUT/'examples.json',all_examples);write(OUT/'summary.json',summary)
    print('SUMMARY',summary['overall'],flush=True)

if __name__=='__main__':main()
