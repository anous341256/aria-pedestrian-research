"""Separate two raw masks for an observed duplicate-identity case; no mask edits."""
import gzip
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from cvat_sdk.masks import decode_mask

ROOT=Path(__file__).resolve().parents[2]
p=argparse.ArgumentParser()
p.add_argument('--recording',default='164557')
p.add_argument('--ids',nargs=2,type=int,default=[3,5])
p.add_argument('--frames',nargs='+',type=int,default=[930,1030,1040,1050])
args=p.parse_args()
BASE=ROOT/'outputs/sam2_fullvideos_20260927'/args.recording
DEST=BASE/'sam2_raw_v1/evidence'
DEST.mkdir(exist_ok=True)
FRAMES=args.frames
IDS=args.ids
masks={}
with gzip.open(BASE/'sam2_raw_v1/masks_rle.jsonl.gz','rt') as handle:
    for line in handle:
        r=json.loads(line)
        if r['frame'] in FRAMES and r['id'] in IDS:
            masks[(r['frame'],r['id'])]=decode_mask(r['rle'],image_width=2560,image_height=1920)
cap=cv2.VideoCapture(str(BASE/'rgb.mp4'))
rows=[];stats=[]
for f in FRAMES:
    cap.set(cv2.CAP_PROP_POS_FRAMES,f);ok,bgr=cap.read();assert ok
    a,b=masks[(f,IDS[0])],masks[(f,IDS[1])]
    intersection=int((a&b).sum());union=int((a|b).sum())
    stats.append({'frame':f,'mask_iou':intersection/union if union else None,f's{IDS[0]}_pixels':int(a.sum()),f's{IDS[1]}_pixels':int(b.sum()),'intersection_pixels':intersection})
    panels=[]
    for oid in [None]+IDS:
        panel=cv2.resize(bgr,(640,480))
        if oid is not None:
            m=cv2.resize(masks[(f,oid)].astype(np.uint8),(640,480),interpolation=cv2.INTER_NEAREST).astype(bool)
            color=np.array({1:(0,220,255),2:(255,180,40),3:(210,60,255),4:(80,230,80),5:(0,120,255)}[oid])
            panel[m]=(panel[m]*.62+color*.38).astype(np.uint8)
        cv2.rectangle(panel,(0,0),(640,38),(0,0,0),-1)
        title='Original RGB' if oid is None else f'Raw S{oid} mask (separate panel)'
        cv2.putText(panel,f'{f/10:.1f}s | {title}',(8,25),cv2.FONT_HERSHEY_SIMPLEX,.56,(255,255,255),1)
        panels.append(panel)
    rows.append(np.hstack(panels))
cap.release()
cv2.imwrite(str(DEST/'duplicate_masks_separated.jpg'),np.vstack(rows),[cv2.IMWRITE_JPEG_QUALITY,94])
(DEST/'duplicate_mask_overlap.json').write_text(json.dumps({'scope':'Pairwise overlap, not accuracy or ground truth. Identity judgement also requires RGB review.','samples':stats},indent=2),encoding='utf-8')
print(json.dumps(stats))
