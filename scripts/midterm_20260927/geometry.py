"""Held-out background correspondence tests, all methods on identical support."""
import argparse
import csv
import sys
from pathlib import Path
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from common import ROOT, OUT, CONFIG, read_json, write_json, apply_h, encoder, finish_encoder
sys.path.insert(0,str(ROOT/'scripts'))
from aria_rectified_rotation_background import background_mask, correspondences, self_test

METHODS=['none','rgb_similarity','rgb_homography','aria_rotation','aria_rgb','inverse_rotation']

def fit_similarity(a,b):
    H=np.eye(3)
    if len(a)<8: return H,False
    A,inliers=cv2.estimateAffinePartial2D(a,b,method=cv2.RANSAC,ransacReprojThreshold=4.,maxIters=2000,confidence=.995,refineIters=10)
    if A is None or not np.isfinite(A).all() or np.mean(inliers)<.3:return H,False
    H[:2]=A
    # Positive, finite and non-explosive estimates. No stale-warp fallback.
    scale=np.sqrt(abs(np.linalg.det(A[:,:2])))
    return (H,True) if .5<scale<2 else (np.eye(3),False)

def fit_homography(a,b):
    H,inliers=cv2.findHomography(a,b,cv2.RANSAC,4.,maxIters=2000,confidence=.995)
    if H is None or not np.isfinite(H).all() or np.mean(inliers)<.3:return np.eye(3),False
    return H/H[2,2],True

def panel(img,a,b,H,title,median):
    out=cv2.resize(img,(640,480));s=640/img.shape[1]
    pred=apply_h(a,H)
    for j in range(0,len(a),max(1,len(a)//45)):
        if not np.isfinite(pred[j]).all(): continue
        p=tuple(np.rint(pred[j]*s).astype(int));q=tuple(np.rint(b[j]*s).astype(int))
        if max(map(abs,p+q))>5000:continue
        cv2.arrowedLine(out,p,q,(20,130,255),1,tipLength=.2)
        cv2.circle(out,q,2,(30,255,100),-1)
    cv2.rectangle(out,(0,0),(640,54),(0,0,0),-1)
    cv2.putText(out,title,(10,22),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1)
    cv2.putText(out,f'held-out median: {median:.1f} px (native)',(10,44),cv2.FONT_HERSHEY_SIMPLEX,.48,(50,230,255),1)
    return out

def run(spec):
    clip=OUT/'clips'/spec['id'];target=clip/'geometry'
    if (target/'summary.json').exists(): print('GEOMETRY REUSE',spec['id'],flush=True);return
    target.mkdir(parents=True,exist_ok=True)
    ref=np.load(clip/'reference.npz');K=ref['K'];Rs=ref['R_world_camera'];ts=ref['times_ns'];xyz=ref['translation_world_device']
    checks=self_test(K,ref['T_device_camera'][:3,:3])
    detections=read_json(clip/'yolo/detections.json')['frames']
    n=len(detections); transforms={m:np.repeat(np.eye(3)[None],n,axis=0) for m in METHODS}
    rgb_valid=np.zeros(n,bool);pose_valid=np.isfinite(Rs).all(axis=(1,2));joint_valid=np.zeros(n,bool)
    rows=[];excluded=[];cap=cv2.VideoCapture(str(clip/'rgb.mp4'));ok,previous=cap.read();assert ok
    g0=cv2.cvtColor(cv2.resize(previous,None,fx=.5,fy=.5),cv2.COLOR_BGR2GRAY)
    mask0=background_mask(g0,np.array(detections[0]).reshape(-1,5),.5)
    proc,log=encoder(target/'comparison.mp4',1280,960,10)
    for f in range(1,n):
        ok,current=cap.read();assert ok
        g1=cv2.cvtColor(cv2.resize(current,None,fx=.5,fy=.5),cv2.COLOR_BGR2GRAY)
        mask1=background_mask(g1,np.array(detections[f]).reshape(-1,5),.5)
        a,b,initial=correspondences(g0,g1,mask0,mask1,.5)
        entry={'frame':f,'source_points':initial,'correspondences':len(a),'status':'valid'}
        if len(a)>=35:
            rng=np.random.default_rng(20260927+f)
            perm=rng.permutation(len(a));cut=int(len(a)*.7);fit=perm[:cut];test=perm[cut:]
            cv2.setRNGSeed(20260927+f)
            Hrgb,rgb_ok=fit_similarity(a[fit],b[fit]); Hh,h_ok=fit_homography(a[fit],b[fit])
            transforms['rgb_similarity'][f]=Hrgb;transforms['rgb_homography'][f]=Hh;rgb_valid[f]=rgb_ok
            if pose_valid[f-1] and pose_valid[f]:
                R=Rs[f].T@Rs[f-1]; Haria=K@R@np.linalg.inv(K);Hinv=K@R.T@np.linalg.inv(K)
                transforms['aria_rotation'][f]=Haria;transforms['inverse_rotation'][f]=Hinv
                aa=apply_h(a[fit],Haria);good=np.isfinite(aa).all(axis=1)
                residual,res_ok=fit_similarity(aa[good],b[fit][good]);Hhybrid=residual@Haria;transforms['aria_rgb'][f]=Hhybrid
                if rgb_ok and h_ok and res_ok:
                    joint_valid[f]=True
                    row={'clip_id':spec['id'],'split':spec['split'],'recording':spec['recording'],'frame':f,
                         'heldout_points':len(test),'fit_points':len(fit),'source_points':initial,'accepted_correspondences':len(a),
                         'rotation_deg':float(np.linalg.norm(Rotation.from_matrix(R).as_rotvec())*180/np.pi),
                         'translation_m':float(np.linalg.norm(xyz[f]-xyz[f-1])),'dt_s':float((ts[f]-ts[f-1])/1e9)}
                    valid_errors=True
                    for m in METHODS:
                        err=np.linalg.norm(apply_h(a[test],transforms[m][f])-b[test],axis=1)
                        if not np.isfinite(err).all():valid_errors=False;break
                        row[m]=float(np.median(err));row[m+'_p90']=float(np.quantile(err,.9))
                    if valid_errors:
                        rows.append(row)
                        tiles=[panel(current,a[test],b[test],transforms[m][f],m,row[m]) for m in ['none','rgb_similarity','aria_rotation','aria_rgb']]
                        img=np.vstack([np.hstack(tiles[:2]),np.hstack(tiles[2:])])
                        if f%20==0:cv2.imwrite(str(target/f'comparison_{f:03d}.jpg'),img)
                    else:entry['status']='nonfinite_projection';joint_valid[f]=False
                else:entry['status']='transform_fit_failed'
            else:entry['status']='pose_invalid'
        else:entry['status']='insufficient_correspondences'
        if entry['status']!='valid' or not joint_valid[f]:
            img=np.tile(cv2.resize(current,(640,480)),(2,2,1))
            cv2.putText(img,entry['status'],(20,35),cv2.FONT_HERSHEY_SIMPLEX,.8,(0,0,255),2)
        proc.stdin.write(img.tobytes());excluded.append(entry)
        g0,mask0=g1,mask1
        if f%25==0:print('GEOMETRY',spec['id'],f,flush=True)
    cap.release();finish_encoder(proc,log)
    np.savez_compressed(target/'transforms.npz',**transforms,rgb_valid=rgb_valid,pose_valid=pose_valid,joint_valid=joint_valid)
    write_json(target/'validity.json',excluded)
    write_json(target/'frame_metrics.json',rows)
    if rows:
        with (target/'frame_metrics.csv').open('w',newline='',encoding='utf-8-sig') as fp:
            wr=csv.DictWriter(fp,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)
    metrics={m:{'median_frame_residual_px':float(np.median([r[m] for r in rows])),
                'median_frame_p90_px':float(np.median([r[m+'_p90'] for r in rows])),
                'improved_vs_none_fraction':float(np.mean([r[m]<r['none'] for r in rows]))} for m in METHODS} if rows else {}
    summary={'spec':spec,'valid_frame_pairs':len(rows),'possible_pairs':n-1,'metrics':metrics,'self_tests':checks,
             'protocol':'KLT independent of Aria, FB <=1px at half resolution, NCC >=0.75; 70% fit / 30% held-out correspondences; same valid support for all methods; no stale warp fallback; masks from automatic person detections',
             'limitation':'Background candidates are not manually verified static GT; errors are against KLT; closed-loop MPS offline reference; no future forecasting claim.'}
    write_json(target/'summary.json',summary)
    print('GEOMETRY COMPLETE',spec['id'],metrics,flush=True)

def aggregate():
    sums=[];rows=[]
    for spec in read_json(CONFIG)['clips']:
        base=OUT/'clips'/spec['id']/'geometry'
        if (base/'summary.json').exists():sums.append(read_json(base/'summary.json'));rows+=read_json(base/'frame_metrics.json')
    if not rows:return
    target=OUT/'results';target.mkdir(exist_ok=True)
    write_json(target/'geometry.json',{'clips':sums,'note':'Aggregate is exploratory; repeated frames are not independent recordings.'})
    fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained')
    for ax,selection,title in [(axes[0],sums,'All frozen clips'),(axes[1],[s for s in sums if s['spec']['split']=='test'],'Held-out recordings')]:
        ms=['none','rgb_similarity','rgb_homography','aria_rotation','aria_rgb','inverse_rotation']
        for s in selection:
            ax.plot(range(len(ms)),[s['metrics'].get(m,{}).get('median_frame_residual_px',np.nan) for m in ms],marker='o',label=s['spec']['id'])
        ax.set_xticks(range(len(ms)),['None','RGB sim.','RGB H','Aria rot.','Aria+RGB','Inverse'],rotation=25)
        ax.set_yscale('log');ax.set_ylabel('Median held-out background error (native px)');ax.set_title(title);ax.grid(alpha=.25);ax.legend(fontsize=7)
    fig.savefig(target/'geometry_comparison.png',dpi=170);plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--clip');p.add_argument('--aggregate-only',action='store_true');a=p.parse_args()
    if not a.aggregate_only:
        for spec in read_json(CONFIG)['clips']:
            if not a.clip or spec['id']==a.clip:run(spec)
    aggregate()
