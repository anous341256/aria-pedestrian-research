"""Compare official SAM2 propagation on three frozen detector-seeded objects per clip.

Partial object coverage is explicit; seed selection does not inspect future output.
This is an annotation proposal, not ground truth or a MOT accuracy benchmark.
"""
import argparse
import importlib.util
import os
from pathlib import Path
import sys
import time
os.environ['HF_HOME']='/opt/cvat-sam2-cache/huggingface'
os.environ['HF_HUB_OFFLINE']='1'
import cv2
import numpy as np
from PIL import Image
import torch
from cvat_sdk.auto_annotation import TrackableShape
from cvat_sdk.masks import decode_mask,encode_mask
from sam2.sam2_image_predictor import SAM2ImagePredictor
from common import ROOT, OUT, CONFIG, read_json, write_json, iou, encoder, finish_encoder
sys.path.insert(0,str(ROOT/'tracking'))
from auto_annotate_people import write_cvat

class Context:
    original_shape_type='mask'

def load_tracker():
    path=ROOT/'tools/cvat-models-official/tracker/sam2/func.py'
    spec=importlib.util.spec_from_file_location('official_sam2_adapter_midterm',path)
    mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
    return mod.create(model_id='facebook/sam2.1-hiera-large',device='cuda')

def run(spec,tracker):
    clip=OUT/'clips'/spec['id'];target=clip/'sam2'
    if (target/'summary.json').exists():print('SAM REUSE',spec['id'],flush=True);return
    target.mkdir(parents=True,exist_ok=True)
    manifest=read_json(clip/'manifest.json');n=manifest['frames'];w,h=manifest['width'],manifest['height']
    dets=read_json(clip/'yolo/detections.json')['frames']
    seeds=[]
    for d in sorted(dets[0],key=lambda d:-d[4]):
        if d[4]<.55 or d[3]-d[1]<80 or d[2]-d[0]<25:continue
        if min(d[0],d[1],w-d[2],h-d[3])<5:continue
        if any(iou(d[:4],s[:4])>.3 for s in seeds):continue
        seeds.append(d)
        if len(seeds)==3:break
    if not seeds:
        write_json(target/'summary.json',{'status':'no_eligible_initial_seed','coverage':'none'});return
    write_json(target/'seeds.json',{'frame':0,'seeds':seeds,'rule':'up to 3 highest score initial-frame person detections: conf>=.55, h>=80,w>=25, no edge truncation, pairwise IoU<=.3',
                                  'source':'automatic detector prompts, no human seed confirmation','scope':'only selected initial-frame people; not all people'})
    yolo=read_json(clip/'yolo/tracks.json')['tracks'];byframe={f:[] for f in range(n)}
    for t in yolo:
        for r in t['boxes']:byframe[r['frame_id']].append(r)
    cap=cv2.VideoCapture(str(clip/'rgb.mp4'));ok,img=cap.read();assert ok
    predictor=SAM2ImagePredictor(tracker._predictor)
    predictor.set_image(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    masks=[]
    for s in seeds:
        m,_,_=predictor.predict(box=np.asarray(s[:4]),multimask_output=False)
        masks.append(m[0].astype(bool))
    del predictor
    pp=tracker.preprocess_image(Context(),Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB)))
    states=[tracker.init_tracking_state(Context(),pp,TrackableShape(type='mask',points=encode_mask(m))) for m in masks]
    del pp
    tracks={k+1:[] for k in range(len(seeds))};mask_rows=[];panels=[]
    proc,log=encoder(target/'comparison.mp4',1280,480,10)
    began=time.monotonic()
    for f in range(n):
        if f:
            ok,img=cap.read();assert ok
            pp=tracker.preprocess_image(Context(),Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB)))
            shapes=[tracker.track(Context(),pp,state) for state in states]
            masks=[decode_mask(s.points,image_width=w,image_height=h) if s else np.zeros((h,w),bool) for s in shapes]
            del pp
        left=cv2.resize(img,(640,480));right=left.copy()
        for r in byframe[f]:
            b=(np.array(r['bbox_xyxy'])*.25).astype(int)
            cv2.rectangle(left,tuple(b[:2]),tuple(b[2:]),(70,240,70),1)
            cv2.putText(left,str(r['track_id']),tuple(b[:2]),cv2.FONT_HERSHEY_SIMPLEX,.36,(70,240,70),1)
        for j,m in enumerate(masks):
            y,x=np.where(m)
            color=[(0,230,255),(255,100,30),(255,50,230)][j]
            if len(x):
                box=[int(x.min()),int(y.min()),int(x.max()+1),int(y.max()+1)]
                tracks[j+1].append({'frame_id':f,'track_id':j+1,'label':'person','bbox_xyxy':box,'confidence':1.0,
                                    'review_status':'automatic','confidence_meaning':'placeholder, not probability'})
                b=(np.array(box)*.25).astype(int)
                small=cv2.resize(m.astype(np.uint8),(640,480),interpolation=cv2.INTER_NEAREST).astype(bool)
                right[small]=(right[small]*.55+np.array(color)*.45).astype(np.uint8)
                cv2.rectangle(right,tuple(b[:2]),tuple(b[2:]),color,2)
                cv2.putText(right,f'S{j+1}',tuple(b[:2]),cv2.FONT_HERSHEY_SIMPLEX,.5,color,1)
            mask_rows.append({'frame':f,'id':j+1,'mask_rle':encode_mask(m),'pixels':int(len(x))})
        for panel,title in [(left,'YOLO11l + BoT-SORT | all proposals'),(right,f'SAM 2 | {len(seeds)} prompted people ONLY')]:
            cv2.rectangle(panel,(0,0),(640,40),(0,0,0),-1)
            cv2.putText(panel,title,(8,17),cv2.FONT_HERSHEY_SIMPLEX,.48,(255,255,255),1)
            cv2.putText(panel,f'{spec["id"]} f={f} AUTO / NOT GOLD',(8,34),cv2.FONT_HERSHEY_SIMPLEX,.40,(0,220,255),1)
        joined=np.hstack([left,right]);proc.stdin.write(joined.tobytes())
        if f%10==0:
            cv2.imwrite(str(target/f'compare_{f:03d}.jpg'),joined)
            panels.append(cv2.resize(joined,(960,360)))
            print('SAM',spec['id'],f,flush=True)
    cap.release();finish_encoder(proc,log)
    video_meta={k:manifest[k] for k in ['width','height','fps']};video_meta['processed_frames']=n
    write_cvat(target/'annotations.xml',tracks,clip/'rgb.mp4',video_meta)
    write_json(target/'tracks.json',{'video':video_meta,'scope':'partial_initial_frame_objects','status':'automatic_not_ground_truth',
                                    'spec':spec,'tracks':[{'track_id':k,'boxes':v} for k,v in tracks.items()]})
    write_json(target/'masks.json',mask_rows)
    cv2.imwrite(str(target/'contact_sheet.jpg'),np.vstack(panels[::2]))
    write_json(target/'summary.json',{'status':'complete_automatic_partial','seeds':len(seeds),'frames':n,
                                     'boxes':sum(map(len,tracks.values())),'seconds':time.monotonic()-began,
                                     'mask_components':'all retained (mask mode, not polygon largest component)',
                                     'limitation':'nonempty mask does not imply identity correctness; visible masks are not amodal whole-body boxes'})
    del states,masks
    torch.cuda.empty_cache()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--clip');a=p.parse_args()
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        tracker=load_tracker()
        for spec in read_json(CONFIG)['clips']:
            if not a.clip or spec['id']==a.clip:run(spec,tracker)
