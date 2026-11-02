"""Near-person offline annotation using frozen V6 propagation and explicit repairs.

Bidirectional propagation and reviewed census are annotation tools, NOT an online
tracker benchmark. Per-frame correction/reinitialization is separately logged.
"""
import os
os.environ['OMP_NUM_THREADS']='2';os.environ['HF_HOME']='/opt/cvat-sam2-cache/huggingface';os.environ['HF_HUB_OFFLINE']='1';os.environ['G1_CLIP']='01'
import sys,importlib.util,time,copy,collections,argparse
from pathlib import Path
import cv2,numpy as np,torch
from PIL import Image
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/sam2_near_people_20261002';SRC=ROOT/'outputs/sam2_g1_v6_v8_20261001'
spec=importlib.util.spec_from_file_location('incremental_v6',ROOT/'scripts/sam2_incremental_v6_20261002/run.py');inc=importlib.util.module_from_spec(spec);spec.loader.exec_module(inc)
from core import *
from common import read_json,write_json,digest
from pilot import load_adapter,Context,decode,describe_mask
from cvat_sdk.auto_annotation import TrackableShape
from cvat_sdk.masks import encode_mask
from sam2.sam2_image_predictor import SAM2ImagePredictor
W,H=2560,1920;CFG=DEFAULTS.copy()

def frame_data(tracker,bgr):
    rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB);pp=tracker.preprocess_image(Context(),Image.fromarray(rgb))
    return rgb,pp,pp.vision_feats[-1][:,0,:].float().cpu().numpy(),cv2.cvtColor(cv2.resize(bgr,(640,480)),cv2.COLOR_BGR2HSV)

def quality(mask,box):
    stats=describe_mask(mask);b=stats['bbox_xyxy'];pix=stats['pixels']
    if b is None or pix<200:return dict(valid=False,reason='empty_or_tiny',box=b,pixels=pix,guide_iou=0.,outside=1.)
    x1,y1,x2,y2=np.array(box);pad=max(12,.10*(y2-y1));x1,y1,x2,y2=np.array([x1-pad,y1-pad,x2+pad,y2+pad]).astype(int);x1=max(0,x1);y1=max(0,y1);x2=min(W,x2);y2=min(H,y2)
    outside=1-mask[y1:y2,x1:x2].sum()/max(1,pix);iou=inc.overlap(b,box)
    valid=iou>=.25 and outside<=.22
    return dict(valid=bool(valid),reason='ok' if valid else 'mask_disagrees_with_reviewed_person_interval',box=b,pixels=pix,guide_iou=iou,outside=float(outside))

def donors_at(archive,f,identity,guide):
    donors=[]
    for key,rows in archive.items():
        if key==identity:continue
        row=rows.get(f)
        if not row or not row.get('donor_eligible'):continue
        mask=decode(row['rle']);pts=inc.interior(mask)
        # Exclude ambiguous points inside the target's central region.
        x1,y1,x2,y2=guide;cx1=x1+.20*(x2-x1);cx2=x2-.20*(x2-x1)
        pts=[p for p in pts if not(cx1<p[0]<cx2 and y1+.1*(y2-y1)<p[1]<y2-.1*(y2-y1))]
        if pts:donors.append(dict(id=key,frame=f,mask=mask,box=row['box'],points=pts,revision=row.get('revision',0)))
    for d in donors:d['points']=[p for p in d['points'] if not any(o['mask'][p[1],p[0]] for o in donors if o is not d)]
    return [d for d in donors if d['points']]

def image_mask(tracker,rgb,box,donors):
    predictor=SAM2ImagePredictor(tracker._predictor);predictor.set_image(rgb)
    neg=inc.near_points(donors,box);kwargs={}
    if neg:kwargs=dict(point_coords=np.array([p['xy'] for p in neg],np.float32),point_labels=np.zeros(len(neg),np.int32))
    masks,scores,_=predictor.predict(box=np.array(box),multimask_output=False,**kwargs)
    return masks[0].astype(bool),float(scores[0]),neg

def init_state(tracker,pp,mask,features,hsv,step,now,Hfrom):
    old=tracker.init_tracking_state(Context(),pp,TrackableShape(type='mask',points=encode_mask(mask)));memory=old.predictor_outputs;memory['cond_frame_outputs']={step:memory['cond_frame_outputs'][0]}
    box=describe_mask(mask)['bbox_xyxy'];anchor=transform_box(box,Hfrom)
    return dict(memory=memory,motion=Motion.initial(anchor,now,step),bank=[inc.base.descriptor(mask,features,hsv)],seed_frame=step,last_reference=step)

def run_segment(tracker,cid,entry,frames,cal,archive):
    dest=OUT/cid/'segments'/entry['segment'];dest.mkdir(parents=True,exist_ok=True)
    if (dest/'runtime.json').exists():return inc.loadgz(dest/'masks_rle.jsonl.gz'),inc.loadgz(dest/'decisions.jsonl.gz')
    write_json(dest/'input.json',entry);begin=time.monotonic();R,K=cal['R'],cal['K'];ki=np.linalg.inv(K);times=cal['times_ns']/1e9
    anchor=entry['anchor_frame'];guides={r['frame']:r for r in entry['rows']};identity=entry['id'];rows=[];logs=[];candidate_logs=[]
    rgb,pp,feat,hsv=frame_data(tracker,frames[anchor]);donors=donors_at(archive,anchor,identity,entry['anchor_box'])
    seed,score,negative=image_mask(tracker,rgb,entry['anchor_box'],donors);q=quality(seed,entry['anchor_box'])
    if not q['valid']:
        write_json(dest/'seed_rejected.json',dict(quality=q,score=score,negative_points=negative));print(cid,entry['segment'],'SEED_REJECT',q,flush=True)
        return [],[dict(frame=anchor,id=identity,segment=entry['segment'],status='SEED_REJECT',quality=q)]
    seeddesc=inc.base.descriptor(seed,feat,hsv)
    rows.append(dict(frame=anchor,id=identity,segment=entry['segment'],batch=entry['batch'],rle=encode_mask(seed),box=q['box'],quality=q,donor_eligible=score>=.7 and q['guide_iou']>=.45,revision=0))
    logs.append(dict(frame=anchor,id=identity,segment=entry['segment'],status='ANCHOR',quality=q,sam_score=score,memory_written=True,negative_points=negative,prior=None))
    del pp,feat,hsv
    for direction in [1,-1]:
        target=list(range(anchor+direction,entry['end']+1 if direction==1 else entry['start']-1,direction))
        if not target:continue
        rgb,pp,feat,hsv=frame_data(tracker,frames[anchor]);state=init_state(tracker,pp,seed,feat,hsv,0,0.,np.eye(3));del pp,feat,hsv
        for step,f in enumerate(target,1):
            guide=guides[f];box=guide['box'];Hto=K@R[f].T@R[anchor]@ki;Hfrom=K@R[anchor].T@R[f]@ki;now=float(abs(times[f]-times[anchor]));state['motion'].predict(now);prior=state['motion'].image_prior(Hto,W,H)
            rgb,pp,feat,hsv=frame_data(tracker,frames[f]);donors=donors_at(archive,f,identity,box)
            candidates=[];evaluations=[]
            for source,prompt in [('memory',None)]+([('motion_box',prior['search_box'])] if prior['search_box'] is not None else []):
                c=inc.corrected_candidate(tracker._predictor,pp,state,step,identity,source,prompt,feat,hsv,donors);e=evaluate_candidate(c,state['motion'],Hfrom,state['bank'],CFG,W,H);qq=quality(c['mask'],box)
                candidate_logs.append(dict(frame=f,step=step,direction=direction,**inc.base.export_candidate(c,e),quality=qq,negative_points=c['negative_points']))
                if not qq['valid']:e=dict(e,valid=False,reasons=e['reasons']+['census_interval_guard'])
                candidates.append(c);evaluations.append(e)
            groups,_=cluster_candidates(candidates,CFG);assignments,_=assign_candidates([identity],candidates,evaluations,groups,CFG);a=assignments[identity]
            mask=None;written=False;negative=[];status='PENDING';info={};q=None
            if a is not None:
                c=candidates[a['candidate']];e=evaluations[a['candidate']];mask=c['mask'];q=quality(mask,box);negative=c['negative_points']
                write,reasons=memory_decision(e,1.,CFG);state['motion'].update(e['innovation']['z'],step,uncertain=not write)
                if write:
                    agreement=0.
                    if len(candidates)==2:
                        m,n=[cc['small_mask'] for cc in candidates];agreement=np.count_nonzero(m&n)/max(1,np.count_nonzero(m|n))
                    promote=step-state['last_reference']>=CFG['keyframe_stride'] and agreement>.55;inc.commit(tracker._predictor,pp,state,c,step,promote);written=True
                    if promote and e['score']>.85:
                        state['bank'].append(c['descriptor'])
                        if len(state['bank'])>3:del state['bank'][1]
                status='V6_TRACKED' if write else 'V6_UNCERTAIN';info=dict(score=e['score'],appearance=e['appearance'],sam_score=c['iou_score'],memory_reason=reasons,selected_candidate=c['candidate_id'])
            # An offline correction is explicit: use this person's reviewed observation
            # box, not an enlarged lost-motion box, and reinitialize only on acceptance.
            if mask is None:
                repaired,sam_score,negative=image_mask(tracker,rgb,box,donors);rq=quality(repaired,box);desc=inc.base.descriptor(repaired,feat,hsv);app=appearance_similarity(desc,[seeddesc])
                if rq['valid'] and sam_score>=.60 and app>=.50:
                    mask=repaired;q=rq;state=init_state(tracker,pp,mask,feat,hsv,step,now,Hfrom);written=True;status='CENSUS_PROMPT_REPAIR';info=dict(sam_score=sam_score,appearance=app,reason='V6 had no acceptable observation; explicit offline correction and memory reset')
                else:
                    state['motion'].missed+=1;status='PENDING_REVIEW';info=dict(sam_score=sam_score,appearance=app,repair_quality=rq)
            if mask is not None:
                donor=written and q['guide_iou']>=.45 and q['outside']<.08 and info.get('sam_score',0)>=.70
                rows.append(dict(frame=f,id=identity,segment=entry['segment'],batch=entry['batch'],rle=encode_mask(mask),box=q['box'],quality=q,donor_eligible=bool(donor),revision=0))
            logs.append(clean(dict(frame=f,id=identity,segment=entry['segment'],direction=direction,step=step,status=status,quality=q,memory_written=written,negative_points=negative,prior=prior,guide=guide,**info)))
            del pp,feat,hsv,candidates,evaluations
        del state;torch.cuda.empty_cache()
    rows.sort(key=lambda r:r['frame']);logs.sort(key=lambda r:r['frame']);inc.dumpgz(dest/'masks_rle.jsonl.gz',rows);inc.dumpgz(dest/'decisions.jsonl.gz',logs);inc.dumpgz(dest/'candidates.jsonl.gz',candidate_logs)
    write_json(dest/'runtime.json',dict(seconds=time.monotonic()-begin,rows=len(rows),expected=entry['end']-entry['start']+1,status='complete',sha256=digest(Path(__file__))))
    print(cid,entry['segment'],f"{len(rows)}/{entry['end']-entry['start']+1}",collections.Counter(d['status'] for d in logs),'sec',round(time.monotonic()-begin,1),flush=True)
    return rows,logs

def main():
    p=argparse.ArgumentParser();p.add_argument('--clips',nargs='+',default=[f'{i:02d}' for i in range(1,11)]);args=p.parse_args()
    write_json(OUT/'protocol.json',dict(base='Frozen V6 motion/scoring/memory functions; per-person bidirectional propagation plus explicit offline census guards and correction resets.',core_sha256=digest(ROOT/'scripts/sam2_g1_v6_v8_20261001/core.py'),near_enter_height_fraction=.12,near_retention_fraction=.08,physical_distance=False,total_person_cap=None,batch_size=3,donor='Only previously completed batches, current-frame mask, strict box agreement and SAM/memory quality. Identity overlap points removed. Own identity never supplies negatives.',annotation='Reviewed proposal aliases and occasional manual boxes; future frames and reference Aria rotations used bidirectionally for offline annotation. Not online tracking or future-only-input prediction evidence.',boundary='Segments limited to reviewed visible intervals. No continuing motion search or negative donation after segment end.',training='Labels are provisional. Short segments kept, not automatically used as forecast training windows.',code_sha256=digest(Path(__file__))))
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        tracker=load_adapter().create(model_id='facebook/sam2.1-hiera-large',device='cuda')
        for cid in args.clips:
            entries=read_json(OUT/cid/'census.json')['segments'];cal=np.load(SRC/cid/'input/reference.npz');cap=cv2.VideoCapture(str(SRC/cid/'input/rgb.mp4'));frames=[]
            while True:
                ok,im=cap.read()
                if not ok:break
                frames.append(im)
            cap.release();archive={};allrows=[];alllogs=[]
            for batch in sorted({e['batch'] for e in entries}):
                pending=[]
                for e in [x for x in entries if x['batch']==batch]:
                    rows,logs=run_segment(tracker,cid,e,frames,cal,archive);allrows+=rows;alllogs+=logs;pending+=rows
                for r in pending:archive.setdefault(r['id'],{})[r['frame']]=r
            inc.dumpgz(OUT/cid/'masks_rle.jsonl.gz',allrows);inc.dumpgz(OUT/cid/'decisions.jsonl.gz',alllogs);write_json(OUT/cid/'complete.json',dict(rows=len(allrows),people=len(archive),segments=len(entries)))
            del frames;print(cid,'COMPLETE',len(allrows),flush=True)
if __name__=='__main__':main()
