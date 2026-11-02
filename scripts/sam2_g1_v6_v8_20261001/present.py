"""Readable Chinese status labels; preserve previous rendering and all raw evidence."""
import sys
from pathlib import Path
import cv2,numpy as np
from render import OUT,ROOT,load,title,BRANCH
from common import read_json,encoder,finish_encoder,write_json
CN={'INITIALIZED':'初始标注','TRACKED':'输出','OBSERVATION_UNCERTAIN':'输出（存疑）','PENDING':'待确认','RECOVERED':'重新确认','LOST':'未输出','UNMATCHED':'未匹配','OUT_OF_VIEW_PRED':'预测出画'}
P=OUT/'presentation';P.mkdir(exist_ok=True)
combined,clog=encoder(P/'all_10_comparison.mp4',1920,880,10,20);count=0;index=[]
for k in range(1,11):
    cid=f'{k:02d}';folder=OUT/cid;logs={v:load(folder/v/'decisions.jsonl.gz') for v in ['v6','v8']}
    meta=read_json(folder/'input/metadata.json');stats=read_json(folder/'comparison/stats.json')
    cap=cv2.VideoCapture(str(folder/'comparison/comparison.mp4'));enc,log=encoder(P/f'{cid}_comparison.mp4',1920,880,10,20);f=0
    while True:
        ok,frame=cap.read()
        if not ok:break
        frame[790:835]=0;lines=[]
        for j,v in enumerate(logs):
            ds=logs[v][f]['branches'][BRANCH]['decisions']
            text='   '.join(f"{i}: {CN[d['status']]}" for i,d in ds.items())
            lines.append((20+j*960,800,text))
        frame=title(frame,lines,23);enc.stdin.write(frame.tobytes());combined.stdin.write(frame.tobytes());count+=1
        if f==stats['sample_frames'][2]:cv2.imwrite(str(P/f'{cid}_preview.jpg'),frame)
        if (cid,f) in [('01',12),('04',20),('04',39),('06',37),('08',25),('09',25),('10',49)]:cv2.imwrite(str(P/f'case_{cid}_{f:03d}.jpg'),frame)
        f+=1
    cap.release();finish_encoder(enc,log);assert f==meta['frames']
    index.append(dict(clip=cid,start_in_montage_seconds=(count-f)/10,duration_seconds=f/10,source=meta['source']['source_video'],source_start_seconds=meta['source']['start_s']))
finish_encoder(combined,clog);assert count==530;write_json(P/'index.json',index)
print('Presentation complete: 530 frames, 53 seconds.',flush=True)
