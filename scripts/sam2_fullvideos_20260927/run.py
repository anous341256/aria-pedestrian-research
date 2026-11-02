"""Five complete recording SAM2 runs with fixed prompted identities and bounded memory."""
import gc
import gzip
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
os.environ['HF_HOME']='/opt/cvat-sam2-cache/huggingface'
os.environ['HF_HUB_OFFLINE']='1'
import cv2
import numpy as np
from PIL import Image
import torch
from cvat_sdk.auto_annotation import TrackableShape
from cvat_sdk.masks import decode_mask,encode_mask
from sam2.sam2_image_predictor import SAM2ImagePredictor

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/midterm_20260927'))
from common import encoder,finish_encoder,write_json,read_json,digest
BASE=ROOT/'outputs/sam2_fullvideos_20260927'
PROMPTS=Path(__file__).with_name('prompts.json')
COLORS={1:(0,220,255),2:(255,180,40),3:(210,60,255),4:(80,230,80),5:(0,120,255)}

class Context:original_shape_type='mask'

def load_file(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;spec.loader.exec_module(m);return m

def gallery(prompts):
    for sid,objects in prompts['recordings'].items():
        tiles=[]
        for obj in objects:
            image=cv2.imread(str(BASE/sid/'preview'/f'{obj["frame"]:05d}.jpg'))
            assert image is not None
            b=obj['box'];color=COLORS[obj['id']]
            cv2.rectangle(image,tuple(b[:2]),tuple(b[2:]),color,4)
            cv2.rectangle(image,(0,0),(1280,50),(0,0,0),-1)
            cv2.putText(image,f'{sid} S{obj["id"]} PROMPT f={obj["frame"]} t={obj["frame"]/10:.1f}s',(12,35),cv2.FONT_HERSHEY_SIMPLEX,1,color,2)
            tiles.append(cv2.resize(image,(640,480)))
        tiles.append(np.zeros_like(tiles[0]))
        cv2.imwrite(str(BASE/sid/'prompt_gallery.jpg'),np.vstack([np.hstack(tiles[j:j+2]) for j in range(0,6,2)]))

def run_one(tracker,previous,sid,objects,prompts):
    clip=BASE/sid;out=clip/'sam2_raw_v1'
    if (out/'summary.json').exists():return read_json(out/'summary.json')
    while not (clip/'manifest.json').exists():
        write_json(BASE/'progress.json',{'stage':'waiting_for_full_export','recording':sid});time.sleep(5)
    out.mkdir(parents=True,exist_ok=True);(out/'review_1fps').mkdir(exist_ok=True);(out/'prompt_masks').mkdir(exist_ok=True)
    manifest=read_json(clip/'manifest.json');n=manifest['frames'];w=manifest['width'];h=manifest['height'];fps=manifest['fps']
    assert len(objects)==5 and all(0<=o['frame']<n for o in objects)
    write_json(out/'prompts_used.json',{'scope':prompts['scope'],'coordinate_size':prompts['coordinate_size'],'objects':objects})
    cap=cv2.VideoCapture(str(clip/'rgb.mp4'));assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT))==n
    proc,log=encoder(out/'comparison.mp4',1920,720,fps,20)
    states={};records={o['id']:[] for o in objects};last={};events=[];tiles=[];page=0;t0=time.monotonic();torch.cuda.reset_peak_memory_stats()
    with gzip.open(out/'masks_rle.jsonl.gz','wt',encoding='utf-8') as handle:
        for f in range(n):
            ok,bgr=cap.read();assert ok,f
            new=[o for o in objects if o['frame']==f]
            masks={}
            if states or new:
                rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
                pp=tracker.preprocess_image(Context(),Image.fromarray(rgb))
                for oid,state in states.items():
                    shape=tracker.track(Context(),pp,state)
                    masks[oid]=decode_mask(shape.points,image_width=w,image_height=h) if shape else np.zeros((h,w),bool)
                if new:
                    predictor=SAM2ImagePredictor(tracker._predictor);predictor.set_image(rgb)
                    for o in new:
                        box=np.asarray(o['box'])*np.array([w/1280,h/960,w/1280,h/960])
                        mask,score,_=predictor.predict(box=box,multimask_output=False)
                        masks[o['id']]=mask[0].astype(bool)
                        assert masks[o['id']].any(),(sid,o['id'],'empty prompt')
                        states[o['id']]=tracker.init_tracking_state(Context(),pp,TrackableShape(type='mask',points=encode_mask(masks[o['id']])))
                    del predictor
                del pp
            right=cv2.resize(bgr,(960,720));left=right.copy()
            for oid,mask in masks.items():
                yy,xx=np.where(mask);pixels=len(xx)
                box=[int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)] if pixels else None
                records[oid].append({'frame':f,'source_rgb_index':f,'bbox_xyxy':box,'mask_pixels':pixels,'human_identity_confirmed':False,'human_box_geometry_confirmed':False,'status':'automatic_unreviewed'})
                handle.write(json.dumps({'frame':f,'id':oid,'rle':encode_mask(mask)},separators=(',',':'))+'\n')
                if oid in last:
                    prior=last[oid]
                    if bool(prior['pixels']) != bool(pixels):events.append({'frame':f,'id':oid,'type':'mask_return' if pixels else 'mask_lost'})
                    elif pixels and (pixels/max(prior['pixels'],1)>4 or prior['pixels']/pixels>4):events.append({'frame':f,'id':oid,'type':'area_change_4x'})
                last[oid]={'pixels':pixels,'box':box};color=COLORS[oid]
                if box:
                    small=cv2.resize(mask.astype(np.uint8),(960,720),interpolation=cv2.INTER_NEAREST).astype(bool)
                    right[small]=(right[small]*.62+np.array(color)*.38).astype(np.uint8)
                    b=(np.array(box)*.375).astype(int);cv2.rectangle(right,tuple(b[:2]),tuple(b[2:]),color,2)
                    cv2.putText(right,f'S{oid}',(max(0,b[0]),max(67,b[1]-5)),cv2.FONT_HERSHEY_SIMPLEX,.65,color,2)
                else:cv2.putText(right,f'S{oid}: no mask',(12,70+oid*22),cv2.FONT_HERSHEY_SIMPLEX,.5,color,1)
            for panel,title in [(left,'Original rectified RGB'),(right,'SAM 2.1 Large | 5 fixed identities')]:
                cv2.rectangle(panel,(0,0),(960,55),(0,0,0),-1)
                cv2.putText(panel,title,(10,22),cv2.FONT_HERSHEY_SIMPLEX,.62,(255,255,255),1)
                cv2.putText(panel,f'{sid} | {f/fps:.1f}/{n/fps:.1f}s | f={f} | AUTOMATIC, NOT GOLD',(10,45),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,220,255),1)
            proc.stdin.write(np.hstack([left,right]).tobytes())
            if new:
                cv2.imwrite(str(out/'prompt_masks'/f'{f:05d}.jpg'),np.hstack([left,right]),[cv2.IMWRITE_JPEG_QUALITY,94])
            if f%10==0 or f==n-1:
                tile=cv2.resize(right,(480,360));tiles.append(tile)
                if len(tiles)==12 or f==n-1:
                    while len(tiles)<12:tiles.append(np.zeros_like(tile))
                    cv2.imwrite(str(out/'review_1fps'/f'page_{page:03d}.jpg'),np.vstack([np.hstack(tiles[j:j+3]) for j in range(0,12,3)]),[cv2.IMWRITE_JPEG_QUALITY,94])
                    page+=1;tiles=[]
            if f%100==0 or f==n-1:
                elapsed=time.monotonic()-t0
                p={'stage':'sam2_running','recording':sid,'frame':f,'frames':n,'elapsed_s':elapsed,'allocated_mib':torch.cuda.memory_allocated()/2**20}
                write_json(BASE/'progress.json',p);print(json.dumps(p),flush=True)
    cap.release();finish_encoder(proc,log);torch.cuda.synchronize()
    tracks=[{'id':o['id'],'description':o['description'],'prompt_frame':o['frame'],'frames':records[o['id']]} for o in objects]
    write_json(out/'tracks.json',{'scope':prompts['scope'],'mask_box_semantics':'visible predicted mask envelope, not amodal full body','tracks':tracks})
    previous.OUT=out;previous.export_boxes(tracks,n,w,h)
    # Correct metadata inherited from the former single-clip serializer.
    import xml.etree.ElementTree as ET
    tree=ET.parse(out/'boxes_unreviewed.xml');tree.find('meta/task/name').text=f'SAM2 full recording {sid}; five prompted people; not gold'
    tree.write(out/'boxes_unreviewed.xml',encoding='utf-8',xml_declaration=True)
    write_json(out/'mechanical_review_events.json',{'note':'Mask state/area changes are review hints, not proven failures.', 'events':events})
    result={'recording':sid,'status':'complete_automatic_partial','full_source_frames':n,'processed_frames':n,'duration_s':n/fps,'objects':5,'object_frame_records':sum(map(len,records.values())),'nonempty_mask_frames':{str(k):sum(r['bbox_xyxy'] is not None for r in rows) for k,rows in records.items()},'nonempty_is_not_accuracy':True,'propagation_seconds_including_render':time.monotonic()-t0,'peak_gpu_allocated_mib':torch.cuda.max_memory_allocated()/2**20,'review_1fps_pages':page,'prompts_sha256':digest(PROMPTS),'script_sha256':digest(Path(__file__)),'comparison_sha256':digest(out/'comparison.mp4'),'rgb_sha256':manifest['video_sha256']}
    write_json(out/'summary.json',result);print('RECORDING COMPLETE',sid,flush=True)
    del states,records,masks,tracks;gc.collect();torch.cuda.empty_cache();return result

def main():
    prompts=read_json(PROMPTS);gallery(prompts)
    if '--gallery-only' in sys.argv:return
    adapter=ROOT/'tools/cvat-models-official/tracker/sam2/func.py';module=load_file('official_sam2_full',adapter)
    previous=load_file('single_clip_serializer',ROOT/'scripts/sam2_scene_trial_20260927/run_sam2.py')
    summaries=[]
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        tracker=module.create(model_id='facebook/sam2.1-hiera-large',device='cuda')
        for sid,objects in prompts['recordings'].items():summaries.append(run_one(tracker,previous,sid,objects,prompts))
    write_json(BASE/'batch_summary.json',{'status':'complete','recordings':summaries,'model':'facebook/sam2.1-hiera-large','adapter_sha256':digest(adapter),'torch':torch.__version__,'inputs':'RGB after factory-calibration rectification and explicit box prompts. No MPS, gaze, detection, later correction or identity reassignment.'})
    write_json(BASE/'progress.json',{'stage':'complete','recordings':5,'frames':sum(x['processed_frames'] for x in summaries)})
    print('BATCH COMPLETE',flush=True)

if __name__=='__main__':main()
