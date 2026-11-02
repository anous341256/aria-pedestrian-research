"""Run one frozen prior-reliability combination on the same five identities."""
import os
os.environ['OMP_NUM_THREADS']='2'
import collections,gzip,json,time
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import torch
from cvat_sdk.auto_annotation import TrackableShape
from cvat_sdk.masks import encode_mask
from sam_io import (ROOT,OLD,VIDEO,POSE,ALIGN,ADAPTER,Camera,descriptor,make_candidate,
    memory_frames,commit_memory,load_adapter,Context,describe_mask,decode,W,H,
    read_json,write_json,digest)
from reliability import (CONFIG,new_state,motion_reliability,prompt_policy,evaluate,adjudicate,
    update_trusted_motion,memory_policy,cluster_candidates,assign_candidates,clean)

OUT=ROOT/'outputs/sam2_reliability_prior_20261001'
BRANCH='motion_with_camera'


def small_overlap(a,b):
    return float(np.count_nonzero(a&b)/max(1,np.count_nonzero(a|b)))


def export_candidate(candidate,evaluation):
    keys=['candidate_id','owner','source','pixels','box','iou_score','presence','prompt_box']
    return clean({**{k:candidate[k] for k in keys},**evaluation})


def run():
    OUT.mkdir(parents=True,exist_ok=True)
    assert not (OUT/'protocol.json').exists(),'Preserve completed or interrupted runs.'
    assert read_json(OUT/'unit_checks.json')['status']=='passed'
    config=CONFIG.copy();previous=read_json(OLD/'protocol.json')
    initial={k:decode(v) for k,v in read_json(OLD/'initial_masks.json').items()}
    camera=Camera(POSE);alignment=read_json(ALIGN)
    assert all(a['status']=='valid' for a in alignment[100:340])
    assert len({a['graph'] for a in alignment[100:340]})==1
    assert np.all(np.diff(camera.times[100:340])>0)
    sources={'rgb':VIDEO,'initial_masks':OLD/'initial_masks.json','pose':POSE,'alignment':ALIGN,'adapter':ADAPTER}
    for v in ['sam2_five_people_20261001','sam2_motion_tracker_v1_20261001','sam2_motion_weight10_20261001']:
        for name in ['protocol.json','masks_rle.jsonl.gz']:
            sources[v+'_'+name]=ROOT/'outputs'/v/name
    protocol=dict(version='reliability_prior_v1',start=100,stop_exclusive=340,fps=10,width=W,height=H,
        people=previous['people'],model=previous['model'],branches=[BRANCH],config=config,
        authorization='User requested implementation of the combined prior and maintenance of earlier attempts.',
        intervention='Combined handcrafted policy, not a single-component causal ablation: historical motion reliability; '
                     'bounded conditional prompts; identity evidence from appearance, competition, continuity and reliable motion; '
                     'pending recovery isolated from primary motion and memory; delayed memory updates.',
        temporal_rule='Reliability and priors computed before current RGB; no future input indices. '
                      'Offline Aria MPS reference provenance does not establish a causal online pose system.',
        source='Aria calibration and offline historical/current reference rotation; not RGB-only.',
        frozen_parameters='Declared before inference. No learned calibration, no full ground truth, no post-run threshold tuning.',
        candidate_vs_output='Tentative candidates stored separately; only policy-confirmed observations output as masks. '
                            'Confirmation is heuristic and may be wrong. No retroactive filling of pending frames.',
        memory='Initial reference retained; at most two extra references and seven recent accepted memories. '
               'Pending candidates never update primary motion, SAM memory or appearance bank.',
        source_hashes={key:dict(path=str(p.relative_to(ROOT)),sha256=digest(p)) for key,p in sources.items()},
        code_hashes={name:digest(Path(__file__).parent/name) for name in ['tracker_geometry.py','sam_io.py','reliability.py','run.py','test_policy.py']})
    write_json(OUT/'protocol.json',protocol)
    states={};summary=collections.Counter();cap=cv2.VideoCapture(str(VIDEO));cap.set(cv2.CAP_PROP_POS_FRAMES,100)
    begin=time.monotonic();torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16), \
         gzip.open(OUT/'decisions.jsonl.gz','wt',encoding='utf-8') as log, \
         gzip.open(OUT/'masks_rle.jsonl.gz','wt',encoding='utf-8') as masks, \
         gzip.open(OUT/'candidates_rle.jsonl.gz','wt',encoding='utf-8') as candidates_out:
        tracker=load_adapter().create(model_id=protocol['model'],device='cuda');model=tracker._predictor
        for f in range(100,340):
            camera.set_current_frame(f);now=float(camera.times[f]);Hto,Hfrom=camera.matrices(f,True)
            priors={};reliabilities={}
            if f>100:
                for i,state in states.items():
                    assert state['motion'].last_observation_frame<f
                    state['motion'].predict(now)
                    priors[i]=state['motion'].image_prior(Hto,W,H)
                    reliabilities[i]=motion_reliability(state,priors[i],Hto,now,config)
            ok,bgr=cap.read();assert ok and bgr.shape[:2]==(H,W)
            pp=tracker.preprocess_image(Context(),Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)))
            features=pp.vision_feats[-1][:,0,:].float().cpu().numpy();assert features.shape[0]==4096
            hsv=cv2.cvtColor(cv2.resize(bgr,(640,480)),cv2.COLOR_BGR2HSV)
            row=dict(frame=f,source_seconds=f/10,time_from_start_seconds=now,current_camera_pose_frame=f,alignment=alignment[f],branches={})
            if f==100:
                decisions={}
                for i,mask in initial.items():
                    tracking=tracker.init_tracking_state(Context(),pp,TrackableShape(type='mask',points=encode_mask(mask)))
                    box=describe_mask(mask)['bbox_xyxy']
                    states[i]=new_state(tracking.predictor_outputs,box,descriptor(mask,features,hsv),now,f)
                    decisions[i]=dict(status='INITIALIZED',reason='reviewed_initial_mask',selected_candidate='initial',
                        tentative_candidate=None,observation_box=box,memory_written=True,reference_promoted=True,
                        memory_after=memory_frames(tracking.predictor_outputs),prior=None,score=None,identity_score=1.,
                        motion_reliability=None,prompt_policy=None,pending_count=0,innovation_px=None,
                        appearance_bank_count=1,motion_updated=True,memory_reason=[])
                    masks.write(json.dumps(dict(frame=f,branch=BRANCH,id=i,rle=encode_mask(mask)))+'\n')
                row['branches'][BRANCH]=dict(decisions=decisions,candidates=[],groups=[],links=[])
            else:
                candidates=[];evaluations=[];policies={};raw_indices={}
                # Generate native-memory candidates first; all identity banks remain frozen within this frame.
                for i,state in states.items():
                    c=make_candidate(model,pp,state,f-100,i,'memory',None,features,hsv)
                    e=evaluate(c,i,states,Hfrom,reliabilities[i],now,config)
                    raw_indices[i]=len(candidates);candidates.append(c);evaluations.append(e)
                for i,state in states.items():
                    raw=evaluations[raw_indices[i]]
                    policy=prompt_policy(state,priors[i],reliabilities[i],raw['identity_score'] if raw['valid'] else 0.,W,H,config)
                    policies[i]=policy;state['prompt_enabled']=policy['enabled']
                    priors[i]['uncontrolled_search_box']=priors[i]['search_box']
                    priors[i]['search_box']=policy['box']
                    if policy['enabled']:
                        c=make_candidate(model,pp,state,f-100,i,'motion_box',policy['box'],features,hsv)
                        e=evaluate(c,i,states,Hfrom,reliabilities[i],now,config)
                        e['raw_candidate_overlap']=small_overlap(c['small_mask'],candidates[raw_indices[i]]['small_mask'])
                        e['conflicts_with_strong_raw']=bool(raw['valid'] and raw['identity_score']>.80 and e['raw_candidate_overlap']<.10)
                        if e['conflicts_with_strong_raw']:e['score']-=.10
                        if e['score']<config['min_match_score']:
                            e['valid']=False
                            if 'low_match_score' not in e['reasons']:e['reasons'].append('low_match_score')
                        candidates.append(c);evaluations.append(e)
                groups,links=cluster_candidates(candidates,config)
                assignments,costs=assign_candidates(list(initial),candidates,evaluations,groups,config)
                decisions={}
                for i,state in states.items():
                    prior=priors[i];r=reliabilities[i];policy=policies[i];assignment=assignments[i]
                    before=memory_frames(state['memory']);bank_before=len(state['bank'])
                    motion_before=state['motion'].x.copy();cov_before=state['motion'].P.copy()
                    old_last=state['motion'].last_observation_frame
                    ci=assignment['candidate'] if assignment else None
                    c=candidates[ci] if ci is not None else None;e=evaluations[ci] if ci is not None else None
                    exploration=bool(c is not None and c['source']=='motion_box' and (policy['exploration'] or e.get('conflicts_with_strong_raw',False)))
                    outcome=adjudicate(state,c,e,f,now,r,exploration,config)
                    decision=dict(prior=prior,motion_reliability=r,prompt_policy=policy,memory_before=before,
                        selected_candidate=None,tentative_candidate=c['candidate_id'] if c is not None else None,
                        assigned_group=assignment['group'] if assignment else None,memory_written=False,reference_promoted=False,
                        observation_box=None,score=e['score'] if e else None,identity_score=e['identity_score'] if e else None,
                        identity_margin=e['identity_margin'] if e else None,innovation_px=None,
                        status=outcome['status'],reason=outcome['reason'],pending_count=outcome['pending_count'],
                        recovery_evidence_frames=[v['frame'] for v in state['pending']],motion_updated=False,
                        memory_reason=['no_confirmed_observation'],appearance_bank_before=bank_before)
                    mask=np.zeros((H,W),bool)
                    if outcome['accepted']:
                        mask=c['mask'];decision['selected_candidate']=c['candidate_id'];decision['observation_box']=c['box']
                        if prior['box'] is not None:
                            observed=(np.asarray(c['box'][:2])+c['box'][2:])/2
                            predicted=(np.asarray(prior['box'][:2])+prior['box'][2:])/2
                            decision['innovation_px']=float(np.linalg.norm(observed-predicted))
                        correction=update_trusted_motion(state,e,f,now,outcome['recovered'],r)
                        decision['motion_correction']=correction;decision['motion_updated']=correction['updated']
                        if not correction['updated']:state['motion'].missed+=1
                        state['last_descriptor']=c['descriptor'];state['identity_confidence']=e['identity_score']
                        state['stable_streak']=1 if outcome['recovered'] else state['stable_streak']+1
                        evidence_good=e['identity_score']>=config['identity_memory'] and e['identity_margin']>=config['identity_margin_memory'] and c['iou_score']>=config['memory_quality'] and c['presence']>=config['memory_presence'] and e['dominant_fraction']>=.70
                        state['memory_streak']=state['memory_streak']+1 if evidence_good and not outcome['recovered'] else int(evidence_good)
                        write,reasons=memory_policy(state,c,e,outcome['recovered'],config)
                        decision['memory_written']=write;decision['memory_reason']=reasons
                        if write:
                            promote=(f-state['last_reference']>=config['keyframe_stride'] and state['stable_streak']>=5 and e['identity_score']>.85)
                            commit_memory(model,pp,state,c,f-100,promote,config)
                            decision['reference_promoted']=promote
                            if promote and e['seed_appearance']>.72:
                                state['bank'].append(c['descriptor'])
                                if len(state['bank'])>3:del state['bank'][1]
                        state['pending']=[]
                    else:
                        state['motion'].missed+=1;state['stable_streak']=0;state['memory_streak']=0
                        state['identity_confidence']*=.9
                        assert np.array_equal(state['motion'].x,motion_before) and np.array_equal(state['motion'].P,cov_before)
                        assert state['motion'].last_observation_frame==old_last
                        if c is None:
                            reasons=sorted({v for candidate,ev in zip(candidates,evaluations) if candidate['owner']==i for v in ev['reasons']})
                            decision['reason']=';'.join(reasons) or 'candidate_claimed_by_another_id'
                    decision.update(memory_after=memory_frames(state['memory']),appearance_bank_count=len(state['bank']),
                        posterior_state=state['motion'].x.copy(),posterior_covariance=state['motion'].P.copy(),
                        last_observation_frame_after=state['motion'].last_observation_frame,
                        missed_frames_after=state['motion'].missed,stable_streak=state['stable_streak'])
                    if not decision['memory_written']:assert before==decision['memory_after'] and bank_before==len(state['bank'])
                    decisions[i]=decision;summary[decision['status']]+=1;summary['memory_writes']+=int(decision['memory_written'])
                    summary['prompts_enabled']+=int(policy['enabled']);summary['prompts_disabled']+=int(not policy['enabled'])
                    masks.write(json.dumps(dict(frame=f,branch=BRANCH,id=i,rle=encode_mask(mask)))+'\n')
                for c in candidates:candidates_out.write(json.dumps(dict(frame=f,branch=BRANCH,candidate_id=c['candidate_id'],rle=encode_mask(c['mask'])))+'\n')
                row['branches'][BRANCH]=dict(decisions=decisions,candidates=[export_candidate(c,e) for c,e in zip(candidates,evaluations)],
                    groups=[[candidates[j]['candidate_id'] for j in g] for g in groups],links=links,assignment_costs=costs)
                del candidates,evaluations
            log.write(json.dumps(clean(row),ensure_ascii=False,allow_nan=False,separators=(',',':'))+'\n')
            del pp,features
            if (f-99)%40==0:print(f'{f-99}/240 | {time.monotonic()-begin:.1f}s | {dict(summary)}',flush=True)
    cap.release();write_json(OUT/'runtime.json',dict(status='inference_complete_pending_review',elapsed_seconds=time.monotonic()-begin,
        peak_gpu_mib=torch.cuda.max_memory_allocated()/2**20,gpu=torch.cuda.get_device_name(),summary=dict(summary),protocol_sha256=digest(OUT/'protocol.json')))
    print('Reliability prior inference complete.',flush=True)


if __name__=='__main__':run()
