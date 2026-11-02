"""Frozen YOLO baseline/detection proposals; no ground-truth claims or auto review."""
import argparse
from collections import defaultdict
from pathlib import Path
import sys
import cv2
import numpy as np
from common import ROOT, OUT, CONFIG, read_json, write_json, encoder, finish_encoder, digest

sys.path.insert(0,str(ROOT/'tracking'))
from auto_annotate_people import write_cvat


def run(spec):
    from ultralytics import YOLO
    import ultralytics,torch
    clip=OUT/'clips'/spec['id']
    target=clip/'yolo'
    if (target/'summary.json').exists(): print('REUSE',spec['id'],flush=True);return
    target.mkdir(parents=True,exist_ok=True)
    meta=read_json(clip/'manifest.json')
    model=YOLO(str(ROOT/'models/tracking/yolo11l.pt'))
    detector=YOLO(str(ROOT/'models/tracking/yolo11l.pt'))
    cap=cv2.VideoCapture(str(clip/'rgb.mp4'))
    tracks=defaultdict(list);detections=[]
    proc,log=encoder(target/'preview.mp4',1280,960,10)
    for frame in range(meta['frames']):
        ok,img=cap.read();assert ok
        # Preserve original initial baseline thresholds; separate raw detections for recall inspection.
        prediction=detector.predict(img,conf=.10,iou=.70,imgsz=1280,classes=[0],verbose=False,device=0)[0]
        detections.append(prediction.boxes.data.cpu().numpy()[:,:5].tolist())
        result=model.track(img,persist=True,conf=.15,iou=.70,imgsz=1280,classes=[0],verbose=False,device=0,
                           tracker=str(ROOT/'tracking/botsort_reid.yaml'))[0]
        overlay=cv2.resize(img,(1280,960))
        if result.boxes is not None and result.boxes.id is not None:
            for tid,box,score in zip(result.boxes.id.int().cpu().tolist(),result.boxes.xyxy.cpu().tolist(),result.boxes.conf.cpu().tolist()):
                tracks[tid].append(dict(frame_id=frame,track_id=tid,label='person',bbox_xyxy=box,confidence=score,review_status='automatic'))
                b=(np.array(box)*.5).astype(int);color=(40+(tid*37)%210,40+(tid*79)%210,40+(tid*111)%210)
                cv2.rectangle(overlay,tuple(b[:2]),tuple(b[2:]),color,2)
                cv2.putText(overlay,f'{tid} {score:.2f}',tuple(b[:2]-[0,5]),cv2.FONT_HERSHEY_SIMPLEX,.45,color,1)
        cv2.rectangle(overlay,(0,0),(1280,45),(0,0,0),-1)
        cv2.putText(overlay,f'{spec["id"]} | frame {frame} | YOLO11l + BoT-SORT | AUTO',(12,28),cv2.FONT_HERSHEY_SIMPLEX,.63,(255,255,255),1)
        proc.stdin.write(overlay.tobytes())
        if frame%20==0:
            cv2.imwrite(str(target/f'frame_{frame:03d}.jpg'),overlay)
            print('YOLO',spec['id'],frame,flush=True)
    cap.release();finish_encoder(proc,log)
    video_meta={k:meta[k] for k in ['width','height','fps']};video_meta['processed_frames']=meta['frames']
    write_cvat(target/'annotations.xml',tracks,clip/'rgb.mp4',video_meta)
    payload={'video':video_meta,'spec':spec,'source_video_sha256':meta['video_sha256'],'status':'automatic_not_ground_truth',
             'method':{'detector':'COCO YOLO11l','tracker_config_sha256':digest(ROOT/'tracking/botsort_reid.yaml'),
                       'conf':.15,'raw_detection_conf':.1,'imgsz':1280,'nms_iou':.7},
             'environment':{'ultralytics':ultralytics.__version__,'torch':torch.__version__},
             'tracks':[{'track_id':tid,'boxes':rows} for tid,rows in sorted(tracks.items())]}
    write_json(target/'tracks.json',payload)
    write_json(target/'detections.json',{'frames':detections,'class':'person','status':'automatic','conf':.1})
    write_json(target/'summary.json',{'frames':meta['frames'],'tracks':len(tracks),'boxes':sum(map(len,tracks.values())),
                                    'continuous_tracks_ge30':sum(len(r)>=30 and r[-1]['frame_id']-r[0]['frame_id']+1==len(r) for r in tracks.values()),
                                    'status':'automatic_not_accuracy'})
    print('YOLO COMPLETE',spec['id'],len(tracks),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--clip');a=p.parse_args()
    for spec in read_json(CONFIG)['clips']:
        if not a.clip or spec['id']==a.clip: run(spec)
