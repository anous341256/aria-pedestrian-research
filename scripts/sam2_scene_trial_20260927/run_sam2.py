"""Uncorrected SAM2.1 Large propagation, explicit prompts, bounded streaming memory."""
import gzip
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
import xml.etree.ElementTree as ET

os.environ['HF_HOME'] = '/opt/cvat-sam2-cache/huggingface'
os.environ['HF_HUB_OFFLINE'] = '1'
import cv2
import numpy as np
from PIL import Image
import torch
from cvat_sdk.auto_annotation import TrackableShape
from cvat_sdk.masks import decode_mask, encode_mask
from sam2.sam2_image_predictor import SAM2ImagePredictor

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'scripts/midterm_20260927'))
from common import encoder, finish_encoder, write_json, read_json, digest
CLIP = ROOT/'outputs/sam2_scene_trial_20260927/162538_020_040'
OUT = CLIP/'sam2_raw_v1'
PROMPTS = Path(__file__).with_name('prompts.json')
COLORS = {1:(0,220,255),2:(255,180,40),3:(210,60,255)}

class Context:
    original_shape_type='mask'

def export_boxes(tracks, n, width, height):
    root=ET.Element('annotations')
    ET.SubElement(root,'version').text='1.1'
    meta=ET.SubElement(root,'meta');task=ET.SubElement(meta,'task')
    ET.SubElement(task,'name').text='SAM2 unreviewed partial trial 162538 20-40s'
    ET.SubElement(task,'size').text=str(n);ET.SubElement(task,'mode').text='interpolation'
    orig=ET.SubElement(task,'original_size')
    ET.SubElement(orig,'width').text=str(width);ET.SubElement(orig,'height').text=str(height)
    labels=ET.SubElement(task,'labels');lab=ET.SubElement(labels,'label')
    ET.SubElement(lab,'name').text='person';ET.SubElement(lab,'color').text='#00ddff'
    for obj in tracks:
        t=ET.SubElement(root,'track',id=str(obj['id']),label='person',source='auto')
        last=None
        for row in obj['frames']:
            box=row['bbox_xyxy']
            if box is None:
                if last is None: continue
                box=last
            attrs=dict(frame=str(row['frame']),outside=str(int(row['bbox_xyxy'] is None)),occluded='0',keyframe='1',xtl=str(box[0]),ytl=str(box[1]),xbr=str(box[2]),ybr=str(box[3]),z_order='0')
            ET.SubElement(t,'box',attrs)
            last=box
    ET.indent(root,space='  ')
    ET.ElementTree(root).write(OUT/'boxes_unreviewed.xml',encoding='utf-8',xml_declaration=True)

def main():
    if (OUT/'summary.json').exists():
        print('Completed result exists; reuse');return
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'frames').mkdir(exist_ok=True);(OUT/'review_sheets').mkdir(exist_ok=True)
    prompts=read_json(PROMPTS);manifest=read_json(CLIP/'manifest.json')
    write_json(OUT/'prompts_used.json',prompts)
    adapter=ROOT/'tools/cvat-models-official/tracker/sam2/func.py'
    spec=importlib.util.spec_from_file_location('official_sam2_trial',adapter)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    t0=time.monotonic()
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        tracker=module.create(model_id='facebook/sam2.1-hiera-large',device='cuda')
        cap=cv2.VideoCapture(str(CLIP/'rgb.mp4'))
        w,h,n=manifest['width'],manifest['height'],manifest['frames']
        proc,log=encoder(OUT/'comparison.mp4',1920,720,10,20)
        states={};records={obj['id']:[] for obj in prompts['objects']}
        tiles=[];page=0
        with gzip.open(OUT/'masks_rle.jsonl.gz','wt',encoding='utf-8') as masks_file:
            for f in range(n):
                ok,bgr=cap.read();assert ok, f
                rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
                pp=tracker.preprocess_image(Context(),Image.fromarray(rgb))
                masks={}
                for oid,state in states.items():
                    shape=tracker.track(Context(),pp,state)
                    masks[oid]=decode_mask(shape.points,image_width=w,image_height=h) if shape else np.zeros((h,w),bool)
                new=[obj for obj in prompts['objects'] if obj['frame']==f]
                if new:
                    predictor=SAM2ImagePredictor(tracker._predictor);predictor.set_image(rgb)
                    for obj in new:
                        mask,score,_=predictor.predict(box=np.asarray(obj['box']),multimask_output=False)
                        masks[obj['id']]=mask[0].astype(bool)
                        states[obj['id']]=tracker.init_tracking_state(Context(),pp,TrackableShape(type='mask',points=encode_mask(masks[obj['id']])))
                    del predictor
                del pp
                right=cv2.resize(bgr,(960,720));left=right.copy()
                for oid,mask in masks.items():
                    yy,xx=np.where(mask)
                    box=[int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)] if len(xx) else None
                    row={'frame':f,'source_rgb_index':200+f,'bbox_xyxy':box,'mask_pixels':int(len(xx)),'human_identity_confirmed':False,'human_box_geometry_confirmed':False,'status':'automatic_unreviewed'}
                    records[oid].append(row)
                    masks_file.write(json.dumps({'frame':f,'id':oid,'rle':encode_mask(mask)},separators=(',',':'))+'\n')
                    color=COLORS[oid]
                    if box:
                        small=cv2.resize(mask.astype(np.uint8),(960,720),interpolation=cv2.INTER_NEAREST).astype(bool)
                        right[small]=(right[small]*.60+np.array(color)*.40).astype(np.uint8)
                        b=(np.array(box)*.375).astype(int)
                        cv2.rectangle(right,tuple(b[:2]),tuple(b[2:]),color,2)
                        cv2.putText(right,f'S{oid}',(max(0,b[0]),max(68,b[1]-5)),cv2.FONT_HERSHEY_SIMPLEX,.7,color,2)
                    else:
                        cv2.putText(right,f'S{oid}: no mask',(12,90+oid*25),cv2.FONT_HERSHEY_SIMPLEX,.55,color,1)
                for panel,title in [(left,'Original rectified RGB'),(right,'SAM 2.1 Large | 3 prompted people ONLY')]:
                    cv2.rectangle(panel,(0,0),(960,57),(0,0,0),-1)
                    cv2.putText(panel,title,(10,23),cv2.FONT_HERSHEY_SIMPLEX,.62,(255,255,255),1)
                    cv2.putText(panel,f'162538 source {20+f/10:.1f}s | local f={f} | AUTO, NOT GOLD',(10,47),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,220,255),1)
                proc.stdin.write(np.hstack([left,right]).tobytes())
                cv2.imwrite(str(OUT/'frames'/f'{f:03d}.jpg'),right,[cv2.IMWRITE_JPEG_QUALITY,90])
                tile=cv2.resize(right,(480,360));tiles.append(tile)
                if len(tiles)==16 or f==n-1:
                    while len(tiles)<16:tiles.append(np.zeros_like(tile))
                    sheet=np.vstack([np.hstack(tiles[j:j+4]) for j in range(0,16,4)])
                    cv2.imwrite(str(OUT/'review_sheets'/f'page_{page:02d}.jpg'),sheet,[cv2.IMWRITE_JPEG_QUALITY,94])
                    page+=1;tiles=[]
                if f%10==0: print('SAM',f,'/200; allocated MiB',round(torch.cuda.memory_allocated()/2**20),flush=True)
        cap.release();finish_encoder(proc,log)
        torch.cuda.synchronize()
    tracks=[{'id':obj['id'],'description':obj['description'],'prompt_frame':obj['frame'],'frames':records[obj['id']]} for obj in prompts['objects']]
    write_json(OUT/'tracks.json',{'scope':prompts['scope'],'mask_box_semantics':'visible mask envelope; not amodal full-body boxes','tracks':tracks})
    export_boxes(tracks,n,w,h)
    write_json(OUT/'summary.json',{'status':'complete_automatic_partial','frames':n,'objects':3,'method':'official CVAT SAM2 adapter, SAM2.1 Large; streaming, mask mode, all components retained','inputs':'RGB and 3 explicit assistant box prompts; no YOLO, MPS, gaze, future corrections or reinitialization','nonempty_mask_frames':{str(k):sum(r['bbox_xyxy'] is not None for r in v) for k,v in records.items()},'evaluated_propagation_frames':{str(k):len(v) for k,v in records.items()},'nonempty_is_not_accuracy':True,'seconds_including_model_load_and_render':time.monotonic()-t0,'peak_gpu_allocated_mib':torch.cuda.max_memory_allocated()/2**20,'torch':torch.__version__,'adapter_sha256':digest(adapter),'script_sha256':digest(Path(__file__)),'prompts_sha256':digest(PROMPTS),'video_sha256':digest(OUT/'comparison.mp4')})
    print('SAM COMPLETE',flush=True)

if __name__=='__main__':
    main()
