"""Evidence sheets for bounded visual review. No implicit label approval."""
import argparse
import cv2
import numpy as np
from common import OUT, CONFIG, read_json, write_json, iou

def runs(rows):
    chunks=[]
    for r in sorted(rows,key=lambda r:r['frame_id']):
        if not chunks or r['frame_id']!=chunks[-1][-1]['frame_id']+1:chunks.append([])
        chunks[-1].append(r)
    return chunks

def build(spec,sam=False):
    clip=OUT/'clips'/spec['id'];target=clip/('samdet_review' if sam else 'review_proposals');target.mkdir(exist_ok=True)
    source=read_json(clip/'yolo/tracks.json')['tracks']
    if sam:
        source=[];dets=read_json(clip/'yolo/detections.json')['frames']
        for t in read_json(clip/'sam2/tracks.json')['tracks']:
            rows=[]
            for r in t['boxes']:
                f=r['frame_id'];ds=[d for d in dets[f] if d[4]>=.15]
                if not ds:continue
                d=max(ds,key=lambda d:iou(d[:4],r['bbox_xyxy']))
                if iou(d[:4],r['bbox_xyxy'])<.3:continue
                rows.append({'frame_id':f,'track_id':t['track_id'],'label':'person','bbox_xyxy':d[:4],'confidence':d[4],'sam_mask_iou':iou(d[:4],r['bbox_xyxy'])})
            source.append({'track_id':t['track_id'],'boxes':rows})
    paths=[]
    for t in source:
        for rs in runs(t['boxes']):
            if len(rs)>=30 and np.median([r['bbox_xyxy'][3]-r['bbox_xyxy'][1] for r in rs])>=80:
                paths.append((t['track_id'],rs))
    paths=sorted(paths,key=lambda x:(-len(x[1]),x[0]))[:3]
    cap=cv2.VideoCapture(str(clip/'rgb.mp4'));frames=[]
    while True:
        ok,im=cap.read()
        if not ok:break
        frames.append(im)
    cap.release();proposals=[]
    for tid,rs in paths:
        key=f'{"samdet" if sam else "yolo"}_{tid}_{rs[0]["frame_id"]}_{rs[-1]["frame_id"]}'
        for start in range(0,len(rs),30):
            tiles=[]
            for r in rs[start:start+30]:
                f=r['frame_id'];b=np.array(r['bbox_xyxy']);im=frames[f].copy();w=b[2]-b[0];h=b[3]-b[1]
                cx,cy=(b[:2]+b[2:])/2;span=max(w*2,h*1.35,220)
                x1=max(0,int(cx-span/2));y1=max(0,int(cy-span/2));x2=min(im.shape[1],int(cx+span/2));y2=min(im.shape[0],int(cy+span/2))
                cv2.rectangle(im,tuple(b[:2].astype(int)),tuple(b[2:].astype(int)),(0,255,255),3)
                tile=cv2.resize(im[y1:y2,x1:x2],(220,220));cv2.rectangle(tile,(0,0),(220,25),(0,0,0),-1)
                cv2.putText(tile,f'ID {tid} f={f}',(7,18),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,255,255),1);tiles.append(tile)
            while len(tiles)<30:tiles.append(np.zeros((220,220,3),np.uint8))
            sheet=np.vstack([np.hstack(tiles[i:i+6]) for i in range(0,30,6)])
            cv2.imwrite(str(target/f'{key}_page{start//30}.jpg'),sheet)
        proposals.append({'key':key,'track_id':tid,'first':rs[0]['frame_id'],'last':rs[-1]['frame_id'],'frames':len(rs),'selection':'three longest contiguous runs >=30 frames and median height >=80px, not selected on forecast gains','source':'SAM2 mask identity + per-frame YOLO detection IoU>=.3' if sam else 'YOLO11l+BoT-SORT','status':'pending_review','boxes':rs})
    write_json(target/'proposals.json',{'clip':spec,'proposals':proposals})
    print(spec['id'],[(p['key'],p['frames']) for p in proposals],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--sam',action='store_true');p.add_argument('--clip');a=p.parse_args()
    for s in read_json(CONFIG)['clips']:
        if not a.clip or s['id']==a.clip:build(s,a.sam)
