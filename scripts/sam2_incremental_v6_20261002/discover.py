"""Exhaustive-frame person proposals for incremental SAM2 annotation, not labels."""
from pathlib import Path
import sys,json,time,collections
import cv2,numpy as np
from ultralytics import YOLO
ROOT=Path(__file__).resolve().parents[2];SRC=ROOT/'outputs/sam2_g1_v6_v8_20261001';OUT=ROOT/'outputs/sam2_incremental_v6_20261002'
sys.path.insert(0,str(ROOT/'scripts/midterm_20260927'))
from common import read_json,write_json,digest
def nms(rows):
    keep=[]
    for row in sorted(rows,key=lambda x:x['confidence'],reverse=True):
        a=np.array(row['box']);ok=True
        for r in keep:
            b=np.array(r['box']);inter=np.prod(np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2])))
            union=np.prod(a[2:]-a[:2])+np.prod(b[2:]-b[:2])-inter
            if inter/max(1,union)>.6:ok=False;break
        if ok:keep.append(row)
    return keep
def run():
    OUT.mkdir(exist_ok=True)
    weights=ROOT/'models/tracking/yolo11l.pt';model=YOLO(str(weights));begin=time.monotonic()
    for k in range(1,11):
        cid=f'{k:02d}';dest=OUT/cid;dest.mkdir(exist_ok=True)
        if (dest/'detections.json').exists():continue
        meta=read_json(SRC/cid/'input/metadata.json');cap=cv2.VideoCapture(str(SRC/cid/'input/rgb.mp4'));frames=[]
        for f in range(meta['frames']):
            ok,im=cap.read();assert ok
            windows=[(0,0,2560,1920)]
            if f%5==0:windows.extend([(0,0,1600,1200),(960,0,2560,1200),(0,720,1600,1920),(960,720,2560,1920)])
            rows=[]
            for j,(x1,y1,x2,y2) in enumerate(windows):
                result=model.predict(im[y1:y2,x1:x2],classes=[0],imgsz=1536 if j==0 else 1280,conf=.12,iou=.65,max_det=1000,device=0,half=True,verbose=False)[0]
                for box,conf in zip(result.boxes.xyxy.cpu().numpy(),result.boxes.conf.cpu().numpy()):
                    box=box+np.array([x1,y1,x1,y1]);rows.append(dict(box=box.tolist(),confidence=float(conf),source='full' if j==0 else f'tile{j}'))
            rows=nms(rows)
            for j,r in enumerate(rows):r['detection_id']=f'{f:03d}_{j:03d}'
            frames.append(dict(frame=f,detections=rows))
        cap.release();write_json(dest/'detections.json',frames)
        print(cid,'frames',len(frames),'proposals',sum(len(r['detections']) for r in frames),'max_frame',max(len(r['detections']) for r in frames),'elapsed',round(time.monotonic()-begin,1),flush=True)
    write_json(OUT/'discovery_protocol.json',dict(model=str(weights),sha256=digest(weights),full_imgsz=1536,tiles_imgsz=1280,tiles_stride=5,confidence=.12,nms=.6,
        scope='All 530 frames from ten selected clips; proposals only. No fixed 3/5-person total limit. max_det=1000 is a detector implementation bound, not target selection.',
        purpose='Find initial and later-entering people; tile passes supplement small-object proposals. Detection is not final mask or ground truth.'))
if __name__=='__main__':run()
