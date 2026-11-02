"""Expand the same night pilot to five fixed people, retaining its intervention."""
import argparse
import gzip
import json
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image
import torch
from cvat_sdk.auto_annotation import TrackableShape
from cvat_sdk.masks import encode_mask
from sam2.sam2_image_predictor import SAM2ImagePredictor

from pilot import (ROOT, OUT as PREVIOUS, W, H, ADAPTER, Context, load_adapter,
                   corrected_step, decode, describe_mask)
from common import read_json, write_json, digest, encoder, finish_encoder

OUT = ROOT / 'outputs/sam2_five_people_20261001'
VIDEO = ROOT / 'outputs/sam2_fullvideos_20260927/201129/rgb.mp4'
COLORS = {'A': (255, 210, 0), 'B': (0, 220, 255), 'C': (75, 230, 100),
          'D': (220, 95, 235), 'E': (0, 130, 255)}
PEOPLE = [
    dict(id='A', old_id=2, box=[652,1178,802,1606], name='Black shirt / gold back print',
         description='黑色短袖、背部金色图案、黑色长裤'),
    dict(id='B', old_id=1, box=[928,1020,1352,1920], name='Gray checked shirt / black bag',
         description='灰格纹长袖衬衫、黑色提包'),
    dict(id='C', old_id=3, box=[822,1180,970,1574], name='White layered dress',
         description='白色层叠短裙/连衣裙'),
    dict(id='D', old_id=None, box=[490,1212,558,1400], name='Light sleeves / long dark outfit',
         description='远处两人中左侧：浅色短袖、深色长款下装'),
    dict(id='E', old_id=None, box=[544,1218,604,1392], name='Light top / shorts / pale bag',
         description='远处两人中右侧：浅色上衣、短裤、浅色袋子'),
]


def put(im, value, xy, color=(238,238,238), scale=.65):
    cv2.putText(im, value, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def display(frame, masks, title, source_frame, width=1280, prompt=None):
    height = width*3//4
    image = cv2.resize(frame, (width,height), interpolation=cv2.INTER_AREA)
    small = {k:cv2.resize(v.astype(np.uint8), (width,height), interpolation=cv2.INTER_NEAREST).astype(bool)
             for k,v in masks.items()}
    count = sum(m.astype(np.uint8) for m in small.values())
    for identity, mask in small.items():
        area = mask & (count == 1)
        image[area] = (.64*image[area]+.36*np.array(COLORS[identity])).astype(np.uint8)
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image, contours, -1, COLORS[identity], 2)
    yy, xx = np.indices((height,width))
    hatch = (count > 1) & ((xx+yy)%12 < 3)
    image[hatch] = (235,235,235)
    anchors=[]
    for identity, mask in small.items():
        if not mask.any():
            continue
        distance=cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
        _,_,_,(x,y)=cv2.minMaxLoc(distance)
        dx={'A':-55, 'B':65, 'C':48, 'D':-50, 'E':48}[identity]
        lx=max(5,min(width-30,x+dx))
        ly=max(24,min(height-12,y-25))
        while any(abs(lx-px)<35 and abs(ly-py)<32 for px,py in anchors):
            ly=min(height-12,ly+35)
            if ly==height-12: break
        anchors.append((lx,ly))
        cv2.line(image,(x,y),(lx+10,ly-8),COLORS[identity],1,cv2.LINE_AA)
        cv2.rectangle(image,(lx-3,ly-23),(lx+26,ly+5),(15,15,15),-1)
        put(image,identity,(lx,ly),COLORS[identity],.85)
    if prompt:
        for x,y in prompt['points']:
            pos=(round(x*width/W),round(y*height/H))
            cv2.circle(image,pos,5,(255,255,255),2,cv2.LINE_AA)
            cv2.circle(image,pos,3,(0,0,255),-1,cv2.LINE_AA)
    canvas=np.full((height+172,width,3),24,np.uint8)
    canvas[48:48+height]=image
    put(canvas,title,(14,30),scale=.70 if width>1000 else .58)
    base=height+48
    present=' '.join(k for k,m in masks.items() if m.any()) or 'none'
    put(canvas,f'Source {source_frame/10:.1f}s | masks: {present} | fixed identities A-E',
        (14,base+23),scale=.59)
    for i,p in enumerate(PEOPLE):
        column=i%2
        row=i//2
        put(canvas,p['id']+': '+p['name'],(14+column*(width//2),base+48+row*24),
            COLORS[p['id']],.51 if width<1100 else .59)
    put(canvas,'White stripes: overlapping IDs | Red points: exclude A from B/C/D/E',
        (14,base+117),scale=.49 if width<1100 else .56)
    return canvas


def seed():
    assert not (OUT/'protocol.json').exists()
    previous=read_json(PREVIOUS/'protocol.json')
    case=next(c for c in previous['cases'] if c['name']=='night_identity')
    protocol=dict(status='initial_masks_pending_assistant_review', model=previous['model'],
                  video=str(VIDEO.relative_to(ROOT)), start=100, stop=340, fps=10,
                  image_size=[W,H], people=PEOPLE, negative_prompts=case['negative_prompts'],
                  variants=['baseline','negative_memory'],
                  method='Five independent object states share image features. A is the donor and uses original propagation. '
                         'B/C/D/E receive the same reviewed A negative points at frames 160,170,190 in the intervention arm. '
                         'No new mutual exclusion across C/D/E; no joint object competition or post-hoc subtraction.',
                  memory='Same pilot policy: initial reference plus latest intervention reference, clear recent automatic memory.',
                  scope='Five fixed initial people in the same 24-second night clip. No identity recycling or manual corrections after seed.',
                  source_sha256=digest(VIDEO), adapter_sha256=digest(ADAPTER),
                  previous_protocol_sha256=digest(PREVIOUS/'protocol.json'))
    cap=cv2.VideoCapture(str(VIDEO))
    cap.set(cv2.CAP_PROP_POS_FRAMES,100)
    ok,bgr=cap.read()
    assert ok
    cap.release()
    rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
    masks={}
    module=load_adapter()
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        tracker=module.create(model_id=protocol['model'],device='cuda')
        predictor=SAM2ImagePredictor(tracker._predictor)
        predictor.set_image(rgb)
        for p in PEOPLE:
            predicted,scores,_=predictor.predict(box=np.array(p['box']),multimask_output=False)
            masks[p['id']]=predicted[0].astype(bool)
            assert masks[p['id']].any()
            p['initial_model_score']=float(scores[0])
    write_json(OUT/'initial_masks.json',{identity:encode_mask(mask) for identity,mask in masks.items()})
    preview=display(bgr,masks,'Five fixed people | initial SAM2 masks',100)
    cv2.imwrite(str(OUT/'initial_five_people.jpg'),preview,[cv2.IMWRITE_JPEG_QUALITY,96])
    close=display(bgr,masks,'Initial masks',100)[48:1008]
    cv2.imwrite(str(OUT/'initial_D_E_crop.jpg'),cv2.resize(close[590:720,225:320],(570,780)),
                [cv2.IMWRITE_JPEG_QUALITY,96])
    write_json(OUT/'protocol.json',protocol)
    print('Five initial masks ready for review.',flush=True)


def run():
    protocol=read_json(OUT/'protocol.json')
    assert protocol['status']=='initial_masks_assistant_reviewed'
    assert digest(ADAPTER)==protocol['adapter_sha256']
    assert not (OUT/'masks_rle.jsonl.gz').exists()
    initial={k:decode(v) for k,v in read_json(OUT/'initial_masks.json').items()}
    module=load_adapter()
    states={}
    records=[]
    audit=[]
    tiles=[]
    page=0
    start=time.monotonic()
    cap=cv2.VideoCapture(str(VIDEO))
    cap.set(cv2.CAP_PROP_POS_FRAMES,protocol['start'])
    single, single_log=encoder(OUT/'five_people.mp4',1280,1132,10,19)
    compare, compare_log=encoder(OUT/'comparison.mp4',1920,892,10,19)
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16),gzip.open(OUT/'masks_rle.jsonl.gz','wt',encoding='utf-8') as handle:
        tracker=module.create(model_id=protocol['model'],device='cuda')
        for f in range(protocol['start'],protocol['stop']):
            ok,bgr=cap.read()
            assert ok and bgr.shape[:2]==(H,W)
            rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
            pp=tracker.preprocess_image(Context(),Image.fromarray(rgb))
            masks={v:{} for v in protocol['variants']}
            prompt=protocol['negative_prompts'].get(str(f))
            for variant in protocol['variants']:
                for identity in initial:
                    if identity=='A' and variant=='negative_memory':
                        masks[variant][identity]=masks['baseline']['A']
                        continue
                    key=(variant,identity)
                    if f==protocol['start']:
                        masks[variant][identity]=initial[identity].copy()
                        states[key]=tracker.init_tracking_state(Context(),pp,
                            TrackableShape(type='mask',points=encode_mask(initial[identity])))
                    elif variant=='negative_memory' and prompt:
                        masks[variant][identity]=corrected_step(tracker,pp,states[key],prompt,
                            use_negative=True,refresh_memory=True)
                        audit.append(dict(frame=f,target=identity,donor='A',points=prompt['points'],
                                          labels=prompt['labels'],passed_to_prompt_encoder=True))
                    else:
                        shape=tracker.track(Context(),pp,states[key])
                        masks[variant][identity]=decode(shape.points) if shape else np.zeros((H,W),bool)
            row=dict(frame=f,intervention=bool(prompt),variants={})
            for variant in protocol['variants']:
                row['variants'][variant]={}
                for identity,mask in masks[variant].items():
                    row['variants'][variant][identity]=describe_mask(mask)
                    handle.write(json.dumps(dict(frame=f,variant=variant,id=identity,rle=encode_mask(mask)),
                                            separators=(',',':'))+'\n')
                row['variants'][variant]['pairwise_overlap_pixels']={}
                for i,identity in enumerate(initial):
                    for other in list(initial)[i+1:]:
                        overlap=int(np.count_nonzero(masks[variant][identity]&masks[variant][other]))
                        row['variants'][variant]['pairwise_overlap_pixels'][identity+other]=overlap
            row['branches_equal']={k:bool(np.array_equal(masks['baseline'][k],masks['negative_memory'][k])) for k in initial}
            if f<160:
                assert all(row['branches_equal'].values())
            records.append(row)
            primary=display(bgr,masks['negative_memory'],'Five people | negative points + memory refresh',f,prompt=prompt)
            left=display(bgr,masks['baseline'],'Five people | original propagation',f,width=960)
            right=display(bgr,masks['negative_memory'],'Five people | negative points + memory refresh',f,width=960,prompt=prompt)
            paired=np.hstack([left,right])
            paired[:,958:962]=240
            single.stdin.write(primary.tobytes())
            compare.stdin.write(paired.tobytes())
            if f%10==0 or f in [181,189,191,193,339]:
                cv2.imwrite(str(OUT/f'five_{f:04d}.jpg'),primary,[cv2.IMWRITE_JPEG_QUALITY,94])
                cv2.imwrite(str(OUT/f'compare_{f:04d}.jpg'),paired,[cv2.IMWRITE_JPEG_QUALITY,94])
            tiles.append(cv2.resize(primary,(640,566)))
            if len(tiles)==8 or f==protocol['stop']-1:
                if len(tiles)%2:tiles.append(np.zeros_like(tiles[-1]))
                sheet=np.vstack([np.hstack(tiles[i:i+2]) for i in range(0,len(tiles),2)])
                cv2.imwrite(str(OUT/f'review_all_{page:03d}.jpg'),sheet,[cv2.IMWRITE_JPEG_QUALITY,92])
                page+=1
                tiles=[]
            del pp
            if (f-protocol['start']+1)%20==0:
                print(f'{f-protocol["start"]+1}/240 frames | {time.monotonic()-start:.1f}s',flush=True)
    cap.release()
    finish_encoder(single,single_log)
    finish_encoder(compare,compare_log)
    write_json(OUT/'records.json',records)
    write_json(OUT/'applied_negative_prompts.json',audit)
    write_json(OUT/'runtime.json',dict(elapsed_seconds=time.monotonic()-start,torch=torch.__version__,
               gpu=torch.cuda.get_device_name(),peak_allocated_gpu_mib=torch.cuda.max_memory_allocated()/2**20,
               script_sha256=digest(Path(__file__)),protocol_sha256=digest(OUT/'protocol.json'),
               frames=len(records),object_masks=len(records)*10,initial_people=5,
               status='inference_complete_pending_review'))
    print('Five-person inference and videos complete.',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['seed','run'])
    globals()[parser.parse_args().action]()
