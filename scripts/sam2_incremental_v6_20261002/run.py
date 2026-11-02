"""Incremental annotation wrapper around the frozen V6 tracker. No person cap."""
import os
os.environ['OMP_NUM_THREADS']='2';os.environ['HF_HOME']='/opt/cvat-sam2-cache/huggingface';os.environ['HF_HUB_OFFLINE']='1';os.environ.setdefault('G1_CLIP','01')
import sys,json,gzip,copy,collections,time,argparse
from pathlib import Path
import cv2,numpy as np,torch
from PIL import Image
ROOT=Path(__file__).resolve().parents[2];SRC=ROOT/'outputs/sam2_g1_v6_v8_20261001';OUT=ROOT/'outputs/sam2_incremental_v6_20261002'
sys.path.insert(0,str(ROOT/'scripts/sam2_g1_v6_v8_20261001'))
import v6_run as base
from core import *
from common import read_json,write_json,digest
from pilot import load_adapter,Context,decode,describe_mask
from cvat_sdk.auto_annotation import TrackableShape
from cvat_sdk.masks import encode_mask
from sam2.sam2_image_predictor import SAM2ImagePredictor
W,H=2560,1920;CONFIG=DEFAULTS.copy()
# Visual review of proposal crop sheets: displays / wearer's phone-hand are not pedestrians.
REJECT={'08':{14:'wearer_hand_phone',24:'wearer_hand_phone',38:'wearer_hand_phone',41:'wearer_hand_phone',75:'wearer_phone',84:'wearer_phone',85:'wearer_hand'},'10':{81:'advertisement_display',82:'advertisement_display',117:'advertisement_display',126:'advertisement_display',135:'advertisement_display'}}
def loadgz(path):
    with gzip.open(path,'rt',encoding='utf8') as f:return [json.loads(x) for x in f]
def dumpgz(path,rows):
    with gzip.open(path,'wt',encoding='utf8') as f:
        for row in rows:f.write(json.dumps(clean(row),ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n')
def overlap(a,b):
    if a is None or b is None:return 0.
    a=np.array(a);b=np.array(b);inter=np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2])).prod();aa=np.maximum(0,a[2:]-a[:2]).prod();bb=np.maximum(0,b[2:]-b[:2]).prod()
    return float(inter/max(1,aa+bb-inter))
def boxmatch(a,b):
    if overlap(a,b)>.35:return True
    if a is None or b is None:return False
    a=np.array(a);b=np.array(b);inter=np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2])).prod();aa=np.prod(a[2:]-a[:2]);bb=np.prod(b[2:]-b[:2])
    center=np.linalg.norm((a[:2]+a[2:]-b[:2]-b[2:])/2)
    return inter/max(1,min(aa,bb))>.78 and center<.22*max(a[3]-a[1],b[3]-b[1])
def interior(mask,count=2):
    small=cv2.resize(mask.astype(np.uint8),(W//2,H//2),interpolation=cv2.INTER_NEAREST)
    dist=cv2.distanceTransform(small,cv2.DIST_L2,5);points=[]
    for _ in range(count):
        _,val,_,loc=cv2.minMaxLoc(dist)
        if val<2:break
        x,y=loc;px,py=x*2,y*2
        if mask[py,px]:points.append([px,py])
        cv2.circle(dist,(x,y),max(8,int(val*2)),0,-1)
    return points
def donors_for_frame(archive,decisions,f,dets):
    donors=[]
    for identity,rows in archive.items():
        row=rows.get(f);d=decisions[identity].get(f)
        if row is None or d is None or not d.get('memory_written') or d.get('observation_box') is None:continue
        if max((overlap(d['observation_box'],r['box']) for r in dets),default=0)<.35:continue
        mask=decode(row['rle']);pts=interior(mask)
        if pts:donors.append(dict(id=identity,mask=mask,box=d['observation_box'],points=pts,frame=f))
    # Never give a point claimed by two donors: overlap is evidence of ambiguity.
    for donor in donors:
        donor['points']=[p for p in donor['points'] if not any(o['mask'][p[1],p[0]] for o in donors if o is not donor)]
    return [d for d in donors if d['points']]
def near_points(donors,box,exclude=None):
    chosen=[]
    if box is None:return chosen
    x1,y1,x2,y2=np.array(box);pad=max(20,.25*(y2-y1));expanded=[x1-pad,y1-pad,x2+pad,y2+pad]
    for d in donors:
        if overlap(expanded,d['box'])<=0:continue
        for p in d['points']:
            if exclude is not None and exclude[p[1],p[0]]:continue
            chosen.append(dict(donor_id=d['id'],frame=d['frame'],xy=p))
    return chosen
def corrected_candidate(model,pp,state,f,identity,source,box,features,hsv,donors):
    candidate=base.make_candidate(model,pp,state,f,identity,source,box,features,hsv)
    negative=near_points(donors,candidate['box'] if candidate['box'] is not None else box)
    raw=dict(box=candidate['box'],pixels=candidate['pixels'],iou_score=candidate['iou_score'],presence=candidate['presence'])
    if negative:
        coords=[];labels=[]
        if box is not None:coords.extend(np.array(box).reshape(2,2).tolist());labels.extend([2,3])
        coords.extend(p['xy'] for p in negative);labels.extend([0]*len(negative))
        coords=np.array(coords,dtype=np.float32)*np.array([model.image_size/W,model.image_size/H],np.float32)
        points=dict(point_coords=torch.as_tensor(coords[None],device='cuda'),point_labels=torch.tensor([labels],dtype=torch.int64,device='cuda'))
        current,sam,_,_=model._track_step(frame_idx=f,is_init_cond_frame=False,current_vision_feats=pp.vision_feats,current_vision_pos_embeds=pp.vision_pos_embeds,feat_sizes=pp.feat_sizes,point_inputs=points,mask_inputs=None,output_dict=state['memory'],num_frames=f+1,track_in_reverse=False,prev_sam_mask_logits=candidate['model_output']['pred_masks'].clamp(-32,32))
        _,_,ious,low,high,pointer,presence=sam
        current.update(pred_masks=low,pred_masks_high_res=high,obj_ptr=pointer,object_score_logits=presence)
        mask=(torch.nn.functional.interpolate(low,size=(H,W),mode='bilinear',align_corners=False,antialias=True)[0,0]>0).cpu().numpy();stats=describe_mask(mask)
        candidate.update(mask=mask,small_mask=cv2.resize(mask.astype(np.uint8),(320,240),interpolation=cv2.INTER_NEAREST).astype(bool),pixels=stats['pixels'],box=stats['bbox_xyxy'],iou_score=float(ious.max()),presence=float(presence.sigmoid().item()),descriptor=base.descriptor(mask,features,hsv),model_output=current,points=points)
    candidate['negative_points']=negative;candidate['before_negative']=raw
    return candidate
def commit(model,pp,state,candidate,f,promote):
    output=candidate['model_output'];model._encode_memory_in_output(pp.vision_feats,pp.feat_sizes,candidate['points'],True,output['pred_masks_high_res'],output['object_score_logits'],output)
    compact=dict(maskmem_features=output['maskmem_features'],maskmem_pos_enc=output['maskmem_pos_enc'][-1:],pred_masks=output['pred_masks'],obj_ptr=output['obj_ptr']);memory=state['memory']
    if promote:
        memory['cond_frame_outputs'][f]=compact;state['last_reference']=f
        while len(memory['cond_frame_outputs'])>1+CONFIG['max_extra_references']:
            old=min(k for k in memory['cond_frame_outputs'] if k!=state['seed_frame']);del memory['cond_frame_outputs'][old]
    else:
        memory['non_cond_frame_outputs'][f]=compact
        while len(memory['non_cond_frame_outputs'])>CONFIG['max_recent_memories']:memory['non_cond_frame_outputs'].popitem(last=False)
def seed_mask(tracker,pp,rgb,entry,donors,known):
    if entry.get('original_id'):
        mask=decode(known[entry['original_id']]);return mask,dict(source='previous_reviewed_initial_mask',negative_points=[])
    predictor=SAM2ImagePredictor(tracker._predictor);predictor.set_image(rgb)
    neg=near_points(donors,entry['box']);kwargs={}
    if neg:kwargs=dict(point_coords=np.array([p['xy'] for p in neg],np.float32),point_labels=np.zeros(len(neg),np.int32))
    masks,scores,_=predictor.predict(box=np.array(entry['box']),multimask_output=False,**kwargs)
    return masks[0].astype(bool),dict(source='person_proposal_box_plus_prior_batch_negative_points',sam_score=float(scores[0]),negative_points=neg)
def run_batch(tracker,cid,batch,entries,archive,logs,dets,enabled=True):
    path=OUT/cid/f'batch_{batch:02d}';path.mkdir(exist_ok=True)
    assert not (path/'runtime.json').exists()
    inp=SRC/cid/'input';cal=np.load(inp/'reference.npz');R,K=cal['R'],cal['K'];ki=np.linalg.inv(K);times=(cal['times_ns']-cal['times_ns'][0])/1e9
    N=len(dets);known=read_json(inp/'initial_masks.json');cap=cv2.VideoCapture(str(inp/'rgb.mp4'));states={};masks=[];decisions=[];candidate_rows=[];seedinfo=[];begin=time.monotonic()
    write_json(path/'input.json',dict(batch=batch,entries=entries,prior_batch_ids=list(archive),negative_enabled=enabled,config=CONFIG))
    for f in range(N):
        Hto=K@R[f].T@R[0]@ki;Hfrom=K@R[0].T@R[f]@ki;now=float(times[f]);priors={}
        for identity,state in states.items():
            assert state['motion'].last_observation_frame<f
            state['motion'].predict(now);priors[identity]=state['motion'].image_prior(Hto,W,H)
        ok,bgr=cap.read();assert ok
        starting=[e for e in entries if e['frame']==f]
        if not states and not starting:continue
        rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB);pp=tracker.preprocess_image(Context(),Image.fromarray(rgb));features=pp.vision_feats[-1][:,0,:].float().cpu().numpy();hsv=cv2.cvtColor(cv2.resize(bgr,(640,480)),cv2.COLOR_BGR2HSV)
        donors=donors_for_frame(archive,logs,f,dets[f]['detections']) if enabled else [];new=[]
        for e in starting:
            identity=e['id'];mask,info=seed_mask(tracker,pp,rgb,e,donors,known);bbox=describe_mask(mask)['bbox_xyxy']
            seedinfo.append(dict(id=identity,frame=f,pixels=int(mask.sum()),box=bbox,**info))
            if mask.sum()<200 or bbox is None:
                decisions.append(dict(frame=f,id=identity,status='SEED_FAILED',observation_box=bbox,memory_written=False,prior=None,negative_points=info['negative_points']));continue
            # The init step has no temporal lookup. Re-index its conditioning output to birth frame.
            old=tracker.init_tracking_state(Context(),pp,TrackableShape(type='mask',points=encode_mask(mask)));memory=old.predictor_outputs
            memory['cond_frame_outputs']={f:memory['cond_frame_outputs'][0]}
            anchorbox=transform_box(bbox,Hfrom);assert anchorbox is not None
            states[identity]=dict(memory=memory,motion=Motion.initial(anchorbox,now,f),bank=[base.descriptor(mask,features,hsv)],last_reference=f,seed_frame=f)
            new.append(identity);masks.append(dict(frame=f,id=identity,batch=batch,rle=encode_mask(mask)))
            decisions.append(dict(frame=f,id=identity,status='INITIALIZED',observation_box=bbox,prior=None,memory_written=True,memory_after=base.memory_frames(memory),negative_points=info['negative_points'],selected_candidate='initial'))
        active=[i for i in states if i not in new];candidates=[];evaluations=[]
        for identity in active:
            state=states[identity];prior=priors[identity];options=[('memory',None)]
            if prior['search_box'] is not None:options.append(('motion_box',prior['search_box']))
            for source,box in options:
                c=corrected_candidate(tracker._predictor,pp,state,f,identity,source,box,features,hsv,donors);e=evaluate_candidate(c,state['motion'],Hfrom,state['bank'],CONFIG,W,H)
                candidates.append(c);evaluations.append(e)
                candidate_rows.append(dict(frame=f,**base.export_candidate(c,e),negative_points=c['negative_points'],before_negative=c['before_negative']))
        if active:
            groups,links=cluster_candidates(candidates,CONFIG);assignments,costs=assign_candidates(active,candidates,evaluations,groups,CONFIG)
            predicted={i:copy.deepcopy(s['motion']) for i,s in states.items()};banks={i:list(s['bank']) for i,s in states.items()}
            for identity in active:
                state=states[identity];prior=priors[identity];a=assignments[identity];before=base.memory_frames(state['memory']);mask=np.zeros((H,W),bool)
                d=dict(frame=f,id=identity,prior=prior,memory_before=before,selected_candidate=None,memory_written=False,reference_promoted=False,observation_box=None,negative_points=[])
                if a is None:
                    owned=[j for j,c in enumerate(candidates) if c['owner']==identity];reasons=sorted({r for j in owned for r in evaluations[j]['reasons']});outside=not prior['projection_valid'] or prior['visible_fraction']<.02
                    d.update(status='OUT_OF_VIEW_PRED' if outside else 'UNMATCHED',reason=';'.join(reasons) or 'candidate_conflict');state['motion'].missed+=1
                else:
                    ci=a['candidate'];c=candidates[ci];e=evaluations[ci];mask=c['mask'];competitors=[]
                    for other in states:
                        if other==identity:continue
                        alt=predicted[other].innovation(c['box'],Hfrom)
                        if alt is not None and alt['mahalanobis2']<9:competitors.append(dict(id=other,appearance=appearance_similarity(c['descriptor'],banks[other]),mahalanobis2=alt['mahalanobis2']))
                    margin=e['appearance']-max((x['appearance'] for x in competitors),default=-1);write,reasons=memory_decision(e,margin,CONFIG)
                    d.update(status='TRACKED' if write else 'OBSERVATION_UNCERTAIN',reason='accepted_visual_observation' if write else ';'.join(reasons),selected_candidate=c['candidate_id'],observation_box=c['box'],score=e['score'],appearance=e['appearance'],identity_margin=margin,memory_written=write,memory_reason=reasons,negative_points=c['negative_points'])
                    state['motion'].update(e['innovation']['z'],f,uncertain=not write)
                    if write:
                        owned=[x for x in candidates if x['owner']==identity and x['pixels']>=200];agreement=0.
                        if len(owned)==2:
                            m,n=[x['small_mask'] for x in owned];agreement=float(np.count_nonzero(m&n)/max(1,np.count_nonzero(m|n)))
                        promote=f-state['last_reference']>=CONFIG['keyframe_stride'] and agreement>.55;commit(tracker._predictor,pp,state,c,f,promote)
                        d['reference_promoted']=promote;d['candidate_agreement_iou']=agreement
                        if promote and e['score']>.85:
                            state['bank'].append(c['descriptor'])
                            if len(state['bank'])>3:del state['bank'][1]
                d['memory_after']=base.memory_frames(state['memory']);d['missed_frames_after']=state['motion'].missed
                assert max(d['memory_after']['references']+d['memory_after']['recent'])<=f
                if not d['memory_written']:assert before==d['memory_after']
                masks.append(dict(frame=f,id=identity,batch=batch,rle=encode_mask(mask)));decisions.append(clean(d))
        del pp,features,candidates,evaluations
    cap.release();dumpgz(path/'masks_rle.jsonl.gz',masks);dumpgz(path/'decisions.jsonl.gz',decisions);dumpgz(path/'candidates.jsonl.gz',candidate_rows);write_json(path/'seeds.json',seedinfo)
    write_json(path/'runtime.json',dict(seconds=time.monotonic()-begin,ids=[e['id'] for e in entries],mask_rows=len(masks),negative_candidate_calls=sum(bool(c['negative_points']) for c in candidate_rows),status='complete',code_sha256=digest(Path(__file__))))
    del states;torch.cuda.empty_cache()
    return masks,decisions
def associate(track,logs):
    votes=collections.Counter()
    for row in track['rows']:
        for identity,frames in logs.items():
            d=frames.get(row['frame'])
            if d and boxmatch(row['box'],d.get('observation_box')):votes[identity]+=1
    if not votes:return None
    identity,hits=votes.most_common(1)[0]
    if hits>=max(2,min(5,int(np.ceil(len(track['rows'])*.3)))):return dict(id=identity,hits=hits,rows=len(track['rows']))
    return None
def add_results(masks,decisions,archive,logs):
    for r in masks:archive.setdefault(r['id'],{})[r['frame']]=r
    for r in decisions:logs.setdefault(r['id'],{})[r['frame']]=r
def run_clip(tracker,cid):
    dest=OUT/cid;dets=read_json(dest/'detections.json');tracks=read_json(dest/'proposal_tracklets.json');archive={};logs={};processed={};entries_all=[];batch=0
    old=read_json(SRC/cid/'input/protocol.json')['people'];pending=[dict(id=p['id'],original_id=p['id'],frame=0,box=p['box_prompt'],source='previous_reviewed_seed') for p in old]
    # Restart only from completed, immutable batches; no output file is overwritten.
    for path in sorted(dest.glob('batch_*/runtime.json')):
        b=path.parent;inp=read_json(b/'input.json');batch=max(batch,inp['batch']);entries_all.extend(inp['entries']);add_results(loadgz(b/'masks_rle.jsonl.gz'),loadgz(b/'decisions.jsonl.gz'),archive,logs)
        for e in inp['entries']:
            if e.get('proposal_id'):processed[e['proposal_id']]=dict(status='attempted',id=e['id'])
        existing={e['id'] for e in inp['entries']};pending=[e for e in pending if e['id'] not in existing]
    next_id=max([len(old)]+[int(e['id'][1:]) for e in entries_all if e['id'].startswith('P')])+1
    while True:
        for t in tracks:
            tid=t['proposal_id']
            if tid in processed:continue
            if tid in REJECT.get(cid,{}):processed[tid]=dict(status='rejected_visual_review',reason=REJECT[cid][tid]);continue
            if not t['eligible']:continue
            association=associate(t,logs)
            if association:processed[tid]=dict(status='associated_existing',**association)
        if not pending:
            todo=sorted([t for t in tracks if t['eligible'] and t['proposal_id'] not in processed],key=lambda t:(t['seed']['frame'],-len(t['rows']),-t['max_conf']))
            for t in todo:
                # Avoid duplicate proposals within the same not-yet-run batch.
                if any(e['frame']==t['seed']['frame'] and boxmatch(e['box'],t['seed']['box']) for e in pending):continue
                s=t['seed'];identity=f'P{next_id:03d}';next_id+=1
                pending.append(dict(id=identity,proposal_id=t['proposal_id'],frame=s['frame'],box=s['box'],source='discovered_person_proposal',confidence=s['confidence']))
                if len(pending)==3:break
        if not pending:break
        chosen=pending[:3];pending=pending[3:];batch+=1
        for e in chosen:
            if e.get('proposal_id'):processed[e['proposal_id']]=dict(status='attempted',id=e['id'])
        print(cid,'batch',batch,[(e['id'],e['frame'],e.get('proposal_id')) for e in chosen],flush=True)
        masks,decisions=run_batch(tracker,cid,batch,chosen,archive,logs,dets);add_results(masks,decisions,archive,logs);entries_all.extend(chosen)
        write_json(dest/'progress.json',dict(batches=batch,initialized_ids=len(entries_all),processed=processed))
    for t in tracks:
        if t['proposal_id'] not in processed:processed[t['proposal_id']]=dict(status='weak_or_short_candidate',reason='fewer_than_3_detections_or_no_confidence_035_height24_seed')
    write_json(dest/'inventory.json',dict(entries=entries_all,proposal_disposition=processed,batches=batch,base_config=CONFIG,identity_note='Proposal-to-existing association is heuristic. IDs are provisional within clips, not unique people across recordings.',unlimited_total=True))
    dumpgz(dest/'masks_rle.jsonl.gz',[r for rows in archive.values() for r in rows.values()]);dumpgz(dest/'decisions.jsonl.gz',[r for rows in logs.values() for r in rows.values()]);print(cid,'COMPLETE',len(entries_all),'IDs',batch,'batches',flush=True)
def main():
    p=argparse.ArgumentParser();p.add_argument('--clips',nargs='+',default=[f'{i:02d}' for i in range(1,11)]);args=p.parse_args()
    write_json(OUT/'protocol.json',dict(batch_size=3,total_person_cap=None,base='V6 original DEFAULTS/scoring/assignment/memory policy; added incremental orchestration and SAM negative correction',core_sha256=digest(ROOT/'scripts/sam2_g1_v6_v8_20261001/core.py'),sources='Same ten rectified G1 clips and Aria current/historical rotations. YOLO used to discover seed boxes, not replace SAM tracking.',negative='Both native-memory and motion-box candidate logits corrected with actual SAM label-0 points. RGB and final masks are never erased/subtracted. Earlier batches only; current frame only.',donor='V6 memory-accepted current mask with current detector IoU >=0.35. Two interior points, minimum 4px interior distance; donor-overlap points excluded. This is a heuristic, not verified identity truth.',new_people='Old reviewed seeds retained, subsequent proposals selected in batches of three. Earliest reliable box in each proposal tracklet initializes SAM. No fixed total person limit.',time='Offline annotation: full-clip proposal persistence and previous-batch coverage are used for discovery and deduplication. Each V6 propagation uses current/past RGB and rotation, not future masks/poses. Not an online detection benchmark.',review='No dense GT. Automatic coverage, overlap and ID counts are diagnostics, not recall, precision, or number of unique physical people. Short weak candidates reviewed separately.',code_sha256=digest(Path(__file__))))
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        tracker=load_adapter().create(model_id='facebook/sam2.1-hiera-large',device='cuda')
        for cid in args.clips:
            if not (OUT/cid/'inventory.json').exists():run_clip(tracker,cid)
if __name__=='__main__':main()
