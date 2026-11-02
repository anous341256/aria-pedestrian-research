"""Run both V1 branches with complete candidate, decision, and memory evidence."""
import os
os.environ['OMP_NUM_THREADS']='2'
os.environ['HF_HOME']='/opt/cvat-sam2-cache/huggingface'
os.environ['HF_HUB_OFFLINE']='1'
import collections
import copy
import gzip
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from PIL import Image
import torch
from cvat_sdk.auto_annotation import TrackableShape
from cvat_sdk.masks import encode_mask

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/sam2_negative_prompts_20261001'))
from pilot import load_adapter,Context,decode,describe_mask,ADAPTER,W,H
from common import read_json,write_json,digest
from core import (DEFAULTS,Motion,appearance_similarity,evaluate_candidate,cluster_candidates,
                  assign_candidates,memory_decision,clean,project)

OUT=ROOT/'outputs/sam2_motion_weight10_20261001'
PARENT=ROOT/'outputs/sam2_motion_tracker_v1_20261001'
OLD=ROOT/'outputs/sam2_five_people_20261001'
VIDEO=ROOT/'outputs/sam2_fullvideos_20260927/201129/rgb.mp4'
POSE=ROOT/'outputs/factor_feasibility_expanded_20260928/201129_reference.npz'
ALIGN=POSE.with_name('201129_alignment.json')
BRANCHES=['motion_no_camera','motion_with_camera']


class Camera:
    def __init__(self,path):
        data=np.load(path)
        self.R=data['R'];self.K=data['K'];self.ki=np.linalg.inv(self.K)
        self.times=(data['times_ns']-data['times_ns'][100])/1e9
        self.anchor=self.R[100].copy()
        self.current_frame=99

    def set_current_frame(self,frame):
        assert frame==self.current_frame+1
        self.current_frame=frame

    def matrices(self,frame,enabled):
        assert 100<=frame<=self.current_frame,'Future camera access is forbidden'
        if not enabled:return np.eye(3),np.eye(3)
        return self.K@self.R[frame].T@self.anchor@self.ki,self.K@self.anchor.T@self.R[frame]@self.ki


def descriptor(mask,features,hsv):
    small=cv2.resize(mask.astype(np.uint8),(256,256),interpolation=cv2.INTER_NEAREST)
    weight=cv2.resize(small.astype(np.float32),(64,64),interpolation=cv2.INTER_AREA).ravel()
    vector=weight@features/max(float(weight.sum()),1e-8)
    vector/=max(float(np.linalg.norm(vector)),1e-8)
    cmask=cv2.resize(mask.astype(np.uint8),(640,480),interpolation=cv2.INTER_NEAREST)
    histogram=cv2.calcHist([hsv],[0,1,2],cmask,[12,4,4],[0,180,0,256,0,256]).ravel()
    histogram=np.sqrt(histogram/max(float(histogram.sum()),1e-8))
    return dict(feature=vector,color=histogram)


def make_candidate(model,pp,state,index,owner,source,search_box,features,hsv):
    points=None
    if search_box is not None:
        coords=np.asarray(search_box,dtype=np.float32).reshape(2,2)
        coords*=np.array([model.image_size/W,model.image_size/H],np.float32)
        points=dict(point_coords=torch.as_tensor(coords[None],device='cuda'),
                    point_labels=torch.tensor([[2,3]],device='cuda',dtype=torch.int64))
    current,sam_outputs,_,_=model._track_step(
        frame_idx=index,is_init_cond_frame=False,current_vision_feats=pp.vision_feats,
        current_vision_pos_embeds=pp.vision_pos_embeds,feat_sizes=pp.feat_sizes,
        point_inputs=points,mask_inputs=None,output_dict=state['memory'],num_frames=index+1,
        track_in_reverse=False,prev_sam_mask_logits=None)
    _,_,ious,low,high,pointer,presence=sam_outputs
    current.update(pred_masks=low,pred_masks_high_res=high,obj_ptr=pointer,object_score_logits=presence)
    mask=(torch.nn.functional.interpolate(low,size=(H,W),mode='bilinear',align_corners=False,
                                          antialias=True)[0,0]>0).cpu().numpy()
    stats=describe_mask(mask)
    result=dict(candidate_id=owner+'_'+source,owner=owner,source=source,mask=mask,
                small_mask=cv2.resize(mask.astype(np.uint8),(320,240),interpolation=cv2.INTER_NEAREST).astype(bool),
                pixels=stats['pixels'],box=stats['bbox_xyxy'],iou_score=float(ious.max()),
                presence=float(presence.sigmoid().item()),prompt_box=search_box,
                descriptor=descriptor(mask,features,hsv),model_output=current,points=points)
    return result


def memory_frames(memory):
    return dict(references=sorted(int(k)+100 for k in memory['cond_frame_outputs']),
                recent=sorted(int(k)+100 for k in memory['non_cond_frame_outputs']))


def commit_memory(model,pp,state,candidate,index,promote,config):
    output=candidate['model_output']
    model._encode_memory_in_output(pp.vision_feats,pp.feat_sizes,candidate['points'],True,
        output['pred_masks_high_res'],output['object_score_logits'],output)
    compact=dict(maskmem_features=output['maskmem_features'],maskmem_pos_enc=output['maskmem_pos_enc'][-1:],
                 pred_masks=output['pred_masks'],obj_ptr=output['obj_ptr'])
    memory=state['memory']
    if promote:
        memory['cond_frame_outputs'][index]=compact
        state['last_reference']=index+100
        while len(memory['cond_frame_outputs'])>1+config['max_extra_references']:
            old=min(k for k in memory['cond_frame_outputs'] if k!=0)
            del memory['cond_frame_outputs'][old]
    else:
        recent=memory['non_cond_frame_outputs']
        recent[index]=compact
        while len(recent)>config['max_recent_memories']:recent.popitem(last=False)


def export_candidate(candidate,evaluation):
    keys=['candidate_id','owner','source','pixels','box','iou_score','presence','prompt_box']
    return clean({**{k:candidate[k] for k in keys},**evaluation})


def run():
    OUT.mkdir(parents=True,exist_ok=True)
    assert not (OUT/'protocol.json').exists(),'Preserve previous attempts; use a new output version.'
    old_protocol=read_json(OLD/'protocol.json')
    initial={k:decode(v) for k,v in read_json(OLD/'initial_masks.json').items()}
    camera=Camera(POSE)
    alignment=read_json(ALIGN)
    assert np.isfinite(camera.R[100:340]).all()
    assert all(a['status']=='valid' for a in alignment[100:340])
    assert len({a['graph'] for a in alignment[100:340]})==1
    assert np.all(np.diff(camera.times[100:340])>0)
    config=DEFAULTS.copy()
    protocol=dict(version='V1_motion_weight_10_percent',start=100,stop_exclusive=340,fps=10,width=W,height=H,
        people=old_protocol['people'],branches=BRANCHES,config=config,model=old_protocol['model'],
        source='Aria offline MPS historical/current reference rotation; not an online pose or RGB-only claim.',
        temporal_rule='Before current RGB measurement, predict using previously accepted observations. '
                      'Use camera at current frame only for coordinate projection. No future frame is accessed.',
        methods='Eight-state constant-velocity Kalman prediction in the initial camera orientation; '
                'raw and motion-box SAM2 observations; masked encoder/color appearance; '
                'duplicate-candidate grouping; exact one-to-one owned-candidate assignment; gated memory writes.',
        scope='Five fixed people, 24-second full clip. Basic conflict handling; no trained re-identification, '
              'depth/translation compensation, true occlusion classifier or long-term recovery.',
        memory='Initial reference plus at most two accepted keyframes; at most seven accepted recent memories. '
               'Rejected observations never enter memory. Predicted masks never enter memory.',
        thresholds='Declared before this V1 run; heuristic, not calibrated correctness probabilities.',
        intervention='Motion score coefficient reduced from .20 to .10; other score coefficients '
                     'scaled proportionally by .90/.80. Prompt-box padding, hard gates, thresholds, '
                     'motion model, assignment and memory rules unchanged. No downstream tuning.',
        parent_run=str(PARENT.relative_to(ROOT)),
        evaluation='Automatic reasons are decision traces, not ground-truth error attribution. '
                   'Candidate innovation is not prediction error. Independent visual review is separate.',
        source_hashes={name:dict(path=str(path.relative_to(ROOT)),sha256=digest(path)) for name,path in
            [('rgb',VIDEO),('initial_masks',OLD/'initial_masks.json'),('pose',POSE),('alignment',ALIGN),
             ('baseline_masks',OLD/'masks_rle.jsonl.gz'),('adapter',ADAPTER),
             ('parent_protocol',PARENT/'protocol.json'),('parent_masks',PARENT/'masks_rle.jsonl.gz'),
             ('parent_decisions',PARENT/'decisions.jsonl.gz')]},
        code_hashes={p.name:digest(p) for p in Path(__file__).parent.glob('*.py')})
    write_json(OUT/'protocol.json',protocol)
    states={branch:{} for branch in BRANCHES}
    counter={branch:collections.Counter() for branch in BRANCHES}
    cap=cv2.VideoCapture(str(VIDEO));cap.set(cv2.CAP_PROP_POS_FRAMES,100)
    begin=time.monotonic();torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16), \
         gzip.open(OUT/'decisions.jsonl.gz','wt',encoding='utf-8') as decision_file, \
         gzip.open(OUT/'masks_rle.jsonl.gz','wt',encoding='utf-8') as masks_file, \
         gzip.open(OUT/'candidates_rle.jsonl.gz','wt',encoding='utf-8') as candidates_file:
        tracker=load_adapter().create(model_id=protocol['model'],device='cuda')
        model=tracker._predictor
        for f in range(100,340):
            camera.set_current_frame(f)
            now=float(camera.times[f])
            priors={};transforms={}
            # Calculate all geometric priors before reading current-frame RGB.
            for branch in BRANCHES:
                transforms[branch]=camera.matrices(f,branch=='motion_with_camera')
                priors[branch]={}
                if f>100:
                    for identity,state in states[branch].items():
                        assert state['motion'].last_observation_frame<f
                        state['motion'].predict(now)
                        priors[branch][identity]=state['motion'].image_prior(transforms[branch][0],W,H)
            ok,bgr=cap.read();assert ok and bgr.shape[:2]==(H,W)
            rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
            pp=tracker.preprocess_image(Context(),Image.fromarray(rgb))
            features=pp.vision_feats[-1][:,0,:].float().cpu().numpy()
            assert features.shape[0]==64*64
            hsv=cv2.cvtColor(cv2.resize(bgr,(640,480)),cv2.COLOR_BGR2HSV)
            frame_log=dict(frame=f,source_seconds=f/10,time_from_start_seconds=now,
                           current_camera_pose_frame=f,alignment=alignment[f],branches={})
            for branch in BRANCHES:
                Hto,Hfrom=transforms[branch]
                if f==100:
                    decisions={}
                    for identity,mask in initial.items():
                        shape=TrackableShape(type='mask',points=encode_mask(mask))
                        old_state=tracker.init_tracking_state(Context(),pp,shape)
                        bbox=describe_mask(mask)['bbox_xyxy']
                        states[branch][identity]=dict(memory=old_state.predictor_outputs,
                            motion=Motion.initial(bbox,now,f),bank=[descriptor(mask,features,hsv)],last_reference=f)
                        decisions[identity]=dict(status='INITIALIZED',reason='reviewed_initial_mask',
                            selected_candidate='initial',observation_box=bbox,memory_written=True,
                            memory_reason=[],reference_promoted=True,memory_after=memory_frames(old_state.predictor_outputs),
                            prior=None,score=None,innovation_px=None)
                        masks_file.write(json.dumps(dict(frame=f,branch=branch,id=identity,rle=encode_mask(mask)))+'\n')
                    frame_log['branches'][branch]=dict(decisions=decisions,candidates=[],groups=[],links=[])
                    continue
                candidates=[];evaluations=[]
                for identity,state in states[branch].items():
                    prior=priors[branch][identity]
                    options=[('memory',None)]
                    if prior['search_box'] is not None:options.append(('motion_box',prior['search_box']))
                    for source,box in options:
                        candidate=make_candidate(model,pp,state,f-100,identity,source,box,features,hsv)
                        evaluation=evaluate_candidate(candidate,state['motion'],Hfrom,state['bank'],config,W,H)
                        candidates.append(candidate);evaluations.append(evaluation)
                        candidates_file.write(json.dumps(dict(frame=f,branch=branch,
                            candidate_id=candidate['candidate_id'],rle=encode_mask(candidate['mask'])),separators=(',',':'))+'\n')
                groups,links=cluster_candidates(candidates,config)
                assignments,costs=assign_candidates(list(initial),candidates,evaluations,groups,config)
                predicted_motions={i:copy.deepcopy(s['motion']) for i,s in states[branch].items()}
                prior_banks={i:list(s['bank']) for i,s in states[branch].items()}
                decisions={}
                for identity,state in states[branch].items():
                    prior=priors[branch][identity]
                    assignment=assignments[identity]
                    before=memory_frames(state['memory'])
                    mask=np.zeros((H,W),bool)
                    decision=dict(prior=prior,memory_before=before,selected_candidate=None,
                                  memory_written=False,reference_promoted=False,score=None,
                                  innovation_px=None,observation_box=None,identity_competitors=[])
                    if assignment is None:
                        owned=[i for i,c in enumerate(candidates) if c['owner']==identity]
                        valid_owned=[i for i in owned if evaluations[i]['valid']]
                        reasons=sorted({reason for i in owned for reason in evaluations[i]['reasons']})
                        if valid_owned:reasons=['candidate_claimed_by_another_id']
                        outside=not prior['projection_valid'] or prior['visible_fraction']<.02
                        decision.update(status='OUT_OF_VIEW_PRED' if outside else 'UNMATCHED',
                            reason=';'.join(reasons) or 'no_usable_mask',memory_reason=['no_accepted_observation'])
                        state['motion'].missed+=1
                    else:
                        ci=assignment['candidate'];candidate=candidates[ci];evaluation=evaluations[ci]
                        mask=candidate['mask']
                        competitors=[]
                        for other,other_state in states[branch].items():
                            if other==identity:continue
                            alternative=predicted_motions[other].innovation(candidate['box'],Hfrom)
                            if alternative is not None and alternative['mahalanobis2']<9:
                                sim=appearance_similarity(candidate['descriptor'],prior_banks[other])
                                competitors.append(dict(id=other,appearance=sim,mahalanobis2=alternative['mahalanobis2']))
                        margin=evaluation['appearance']-max((c['appearance'] for c in competitors),default=-1)
                        write,reasons=memory_decision(evaluation,margin,config)
                        observation_center=(np.array(candidate['box'][:2])+candidate['box'][2:])/2
                        predicted_center=(np.array(prior['box'][:2])+prior['box'][2:])/2 if prior['box'] is not None else None
                        delta=None if predicted_center is None else float(np.linalg.norm(observation_center-predicted_center))
                        decision.update(status='TRACKED' if write else 'OBSERVATION_UNCERTAIN',
                            reason='accepted_visual_observation' if write else ';'.join(reasons),
                            selected_candidate=candidate['candidate_id'],selected_group=assignment['group'],
                            observation_box=candidate['box'],score=evaluation['score'],appearance=evaluation['appearance'],
                            innovation_px=delta,mahalanobis2=evaluation['innovation']['mahalanobis2'],
                            identity_margin=margin,identity_competitors=competitors,memory_written=write,memory_reason=reasons)
                        state['motion'].update(evaluation['innovation']['z'],f,uncertain=not write)
                        if write:
                            owned=[c for c in candidates if c['owner']==identity and c['pixels']>=200]
                            agreement=0.
                            if len(owned)==2:
                                a,b=[c['small_mask'] for c in owned]
                                agreement=float(np.count_nonzero(a&b)/max(1,np.count_nonzero(a|b)))
                            promote=f-state['last_reference']>=config['keyframe_stride'] and agreement>.55
                            commit_memory(model,pp,state,candidate,f-100,promote,config)
                            decision['reference_promoted']=promote
                            decision['candidate_agreement_iou']=agreement
                            if promote and evaluation['score']>.85:
                                state['bank'].append(candidate['descriptor'])
                                if len(state['bank'])>3:del state['bank'][1]
                    decision['memory_after']=memory_frames(state['memory'])
                    if not decision['memory_written']:assert before==decision['memory_after']
                    decision['posterior_state']=state['motion'].x.copy()
                    decision['posterior_covariance']=state['motion'].P.copy()
                    decision['missed_frames_after']=state['motion'].missed
                    decisions[identity]=decision
                    counter[branch][decision['status']]+=1
                    counter[branch]['memory_writes']+=int(decision['memory_written'])
                    masks_file.write(json.dumps(dict(frame=f,branch=branch,id=identity,rle=encode_mask(mask)),separators=(',',':'))+'\n')
                frame_log['branches'][branch]=dict(decisions=decisions,
                    candidates=[export_candidate(c,e) for c,e in zip(candidates,evaluations)],
                    groups=[[candidates[i]['candidate_id'] for i in group] for group in groups],links=links,
                    assignment_costs=costs)
                del candidates,evaluations
            decision_file.write(json.dumps(clean(frame_log),ensure_ascii=False,allow_nan=False,separators=(',',':'))+'\n')
            del pp,features
            if (f-99)%20==0:
                print(f'{f-99}/240 frames | {time.monotonic()-begin:.1f}s | '+
                      ' | '.join(branch+': '+str(dict(counter[branch])) for branch in BRANCHES),flush=True)
    cap.release()
    write_json(OUT/'runtime.json',dict(status='inference_complete_pending_render_and_review',
        elapsed_seconds=time.monotonic()-begin,torch=torch.__version__,gpu=torch.cuda.get_device_name(),
        peak_gpu_mib=torch.cuda.max_memory_allocated()/2**20,summary={k:dict(v) for k,v in counter.items()},
        protocol_sha256=digest(OUT/'protocol.json')))
    print('V1 inference complete.',flush=True)


if __name__=='__main__':run()
