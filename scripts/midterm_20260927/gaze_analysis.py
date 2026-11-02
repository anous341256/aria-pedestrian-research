"""Exploratory gaze/person analysis: no causal claim, no learned gaze model."""
import argparse
import csv
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import OUT, CONFIG, read_json, write_json, apply_h, encoder, finish_encoder

def hit(p,b):return bool(np.isfinite(p).all() and b[0]<=p[0]<=b[2] and b[1]<=p[1]<=b[3])
def center(b):return (np.asarray(b[:2])+b[2:])/2
def summarize(spec,reviewed=False,*,tracks_file=None,output_subdir=None,identity_source=None):
    clip=OUT/'clips'/spec['id'];out=clip/(output_subdir or ('gaze_reviewed' if reviewed else 'gaze'));out.mkdir(exist_ok=True)
    ref=np.load(clip/'reference.npz');audit=read_json(clip/'alignment_audit.json')['gaze'];n=len(audit)
    g=ref['gaze_xy_direction'];gs=ref['gaze_xy_spatial'];geo=np.load(clip/'geometry/transforms.npz')
    det=read_json(clip/'yolo/detections.json')['frames'];tracks=read_json(tracks_file or clip/('annotation_v1' if reviewed else 'yolo')/'tracks.json')['tracks']
    selected={}
    for a in audit:
        if a['status']!='valid_direction':continue
        ts=a['gaze_sample_timestamp_ns'];prev=selected.get(ts)
        if prev is None or a['gap_ms']<audit[prev]['gap_ms']:selected[ts]=a['frame']
    fs=sorted(selected.values())
    byframe=[{} for _ in range(n)];residual={}
    for tr in tracks:
        for r in tr['boxes']:byframe[r['frame_id']][tr['track_id']]=r['bbox_xyxy']
    for f in range(1,n):
        if not geo['joint_valid'][f]:continue
        for tid,b in byframe[f].items():
            if tid not in byframe[f-1]:continue
            p=center(byframe[f-1][tid]);pred=apply_h([p],geo['rgb_homography'][f])[0]
            residual[f,tid]=center(b)-pred
    rows=[];lags=np.arange(-20,21,5);lag_deltas={str(int(l)):[] for l in lags}
    for f in fs:
        boxes=[d[:4] for d in det[f] if d[4]>=.15];row={'frame':f,'gaze_timestamp_ns':audit[f]['gaze_sample_timestamp_ns'],
          'gap_ms':audit[f]['gap_ms'],'person_detections':len(boxes),'direction_hits_person':any(hit(g[f],b) for b in boxes),
          'center_hits_person':any(hit(np.array([1279.5,959.5]),b) for b in boxes),'spatial_valid':bool(np.isfinite(gs[f]).all()),
          'spatial_hits_person':any(hit(gs[f],b) for b in boxes)}
        for shift in [-20,-10,10,20]:
            # Circular within-clip shift keeps gaze values/marginal distribution; diagnostic, not independent controls.
            row[f'shift_{shift/10:g}s_hits_person']=any(hit(g[(f+shift)%n],b) for b in boxes)
        rows.append(row)
        for lag in lags:
            groups=[[],[]]
            for tid,b in byframe[f].items():
                k=f+int(lag)
                if (k,tid) not in residual:continue
                # Require uninterrupted identity observations between gaze and speed time.
                if any(tid not in byframe[q] for q in range(max(0,min(f,k)-1),min(n,max(f,k)+1))):continue
                v=float(np.linalg.norm(residual[k,tid])*10/2560)
                groups[int(hit(g[f],b))].append(v)
            if all(groups):lag_deltas[str(int(lag))].append(float(np.mean(groups[1])-np.mean(groups[0])))
    spatial_rows=[r for r in rows if r['spatial_valid']]
    summary={'spec':spec,'unique_gaze_samples':len(fs),'video_frames':n,'native_gaze_rate_hz':5,
      'motion_identity_source':identity_source or ('assistant_prechecked_partial_not_gold' if reviewed else 'automatic_full_proposals'),
      'spatial_common_support':{'samples':len(spatial_rows),'direction_hit':float(np.mean([r['direction_hits_person'] for r in spatial_rows])) if spatial_rows else None,
                               'spatial_hit':float(np.mean([r['spatial_hits_person'] for r in spatial_rows])) if spatial_rows else None},
      'max_matching_error_ms':max((r['gap_ms'] for r in rows),default=None),
      'hit_rates':{key:float(np.mean([r[key] for r in rows])) for key in rows[0] if key.endswith('hits_person')},
      'lag_motion_difference':{k:{'matched_times':len(v),'mean_gazed_minus_other_image_width_per_s':float(np.mean(v)) if v else None} for k,v in lag_deltas.items()},
      'meaning':'Positive lag = later person residual speed; same-frame gazed minus other tracks. Exploratory automatic-label relationship, NOT causal or predictive validation.',
      'limitations':['general gaze without user calibration','direction projection ignores CPF/RGB origin baseline','5Hz sampling; no sub-200ms timing conclusion','automatic identities can switch','boxes not head/face/eye contact labels','circular time-shift controls are within-clip diagnostics, not independent samples','RGB background residual includes parallax and observation noise, not true metric pedestrian speed']}
    write_json(out/'summary.json',summary);write_json(out/'sample_rows.json',rows)
    with (out/'sample_rows.csv').open('w',newline='',encoding='utf-8-sig') as fp:
        wr=csv.DictWriter(fp,fieldnames=rows[0]);wr.writeheader();wr.writerows(rows)
    if not reviewed and not (out/'preview.mp4').exists():
        cap=cv2.VideoCapture(str(clip/'rgb.mp4'));proc,log=encoder(out/'preview.mp4',1280,960,10)
        for f in range(n):
            ok,im=cap.read();assert ok;im=cv2.resize(im,(1280,960))
            for tid,b in byframe[f].items():
                b=np.asarray(b);color=(0,255,255) if hit(g[f],b) else (90,150,90)
                cv2.rectangle(im,tuple((b[:2]/2).astype(int)),tuple((b[2:]/2).astype(int)),color,1)
                if (f,tid) in residual:
                    q=center(b)/2;start=q-residual[f,tid]/2
                    cv2.arrowedLine(im,tuple(start.astype(int)),tuple(q.astype(int)),(0,140,255),2,tipLength=.25)
            if np.isfinite(g[f]).all():cv2.drawMarker(im,tuple((g[f]/2).astype(int)),(255,0,255),cv2.MARKER_CROSS,26,3)
            if np.isfinite(gs[f]).all():cv2.circle(im,tuple((gs[f]/2).astype(int)),10,(255,230,0),2)
            cv2.rectangle(im,(0,0),(1280,72),(0,0,0),-1)
            for j,line in enumerate([f'{spec["id"]}  f={f} | EXPLORATORY / AUTOMATIC TRACKS',
                'Magenta: gaze direction; cyan: valid spatial gaze; yellow box: gaze intersects person',
                'Orange arrow: observed person displacement remaining after RGB background homography']):
                cv2.putText(im,line,(12,20+j*23),cv2.FONT_HERSHEY_SIMPLEX,.50,(255,255,255),1)
            proc.stdin.write(im.tobytes())
            if f%20==0:cv2.imwrite(str(out/f'frame_{f:03d}.jpg'),im)
        cap.release();finish_encoder(proc,log)
    print('GAZE',spec['id'],summary['unique_gaze_samples'],summary['hit_rates'],flush=True)
    return summary

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--reviewed',action='store_true');a=p.parse_args()
    sums=[summarize(s,a.reviewed) for s in read_json(CONFIG)['clips']]
    write_json(OUT/f'results/gaze{"_reviewed" if a.reviewed else ""}.json',sums)
    fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained')
    keys=['direction_hits_person','center_hits_person','shift_-2s_hits_person','shift_2s_hits_person'];x=np.arange(len(sums))
    for j,key in enumerate(keys):axes[0].bar(x+(j-1.5)*.20,[s['hit_rates'][key] for s in sums],.2,label=key)
    axes[0].set_xticks(x,[s['spec']['id'] for s in sums],rotation=40,ha='right');axes[0].set_ylabel('Gaze/center inside any detected person');axes[0].legend(fontsize=7);axes[0].set_ylim(0,1)
    for s in sums:
        data=s['lag_motion_difference'];axes[1].plot([int(k)/10 for k in data],[v['mean_gazed_minus_other_image_width_per_s'] for v in data.values()],marker='.',label=s['spec']['id'])
    axes[1].axhline(0,c='gray',lw=1);axes[1].set_xlabel('Lag seconds (positive = later person motion)');axes[1].set_ylabel('Gazed minus other residual speed (image width/s)');axes[1].legend(fontsize=7)
    fig.suptitle('Exploratory gaze | partial prechecked tracks' if a.reviewed else 'Exploratory general-gaze association | automatic tracks, not causal evidence');fig.savefig(OUT/f'results/gaze{"_reviewed" if a.reviewed else ""}_comparison.png',dpi=170);plt.close(fig)
