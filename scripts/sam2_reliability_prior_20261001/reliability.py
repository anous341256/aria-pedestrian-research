"""Handcrafted reliability and recovery policy. Scores are not calibrated probabilities."""
from collections import deque
import math
import cv2
import numpy as np
from tracker_geometry import (Motion,transform_box,box_to_z,clip_box,appearance_similarity,
                      cluster_candidates,assign_candidates,clean)

CONFIG=dict(min_iou=.40,min_presence=.35,min_match_score=.64,
    cluster_iou=.65,cluster_containment=.88,cluster_area_ratio=.25,
    max_recent_memories=7,max_extra_references=2,keyframe_stride=10,
    missing_tau_seconds=.7,uncertainty_scale=.50,residual_scale=.75,
    prompt_on=.45,prompt_keep=.30,prompt_explore=.65,
    max_prompt_area=.40,max_prompt_relative_area=4.,padding_scale_limit=.65,
    identity_accept=.72,identity_candidate=.52,identity_memory=.80,
    identity_margin_accept=-.04,identity_margin_memory=.015,
    recovery_frames=3,long_gap_frames=10,long_recovery_frames=5,
    memory_streak=3,memory_quality=.65,memory_presence=.65,
    max_pending_gap_frames=1,continuity_distance=1.25,continuity_log_scale=.65)


def new_state(memory,box,desc,now,frame):
    return dict(memory=memory,motion=Motion.initial(box,now,frame),bank=[desc],
        last_descriptor=desc,last_anchor_box=np.asarray(box,dtype=float),last_trusted_time=now,
        last_reference=frame,identity_confidence=1.,stable_streak=1,memory_streak=1,
        pending=[],residuals=deque(maxlen=5),prompt_enabled=True)


def motion_reliability(state,prior,to_current,now,config=CONFIG):
    trusted_box=transform_box(state['last_anchor_box'],to_current)
    valid=trusted_box is not None and prior['projection_valid'] and prior['sigma_xy'] is not None
    gap=max(0.,now-state['last_trusted_time']-.100001)
    if not valid:
        return dict(score=0.,geometry_valid=False,missing_seconds=gap,uncertainty_ratio=None,
                    residual=None,reference_box=trusted_box,history_factor=0.)
    size=np.maximum(np.asarray(trusted_box[2:])-trusted_box[:2],8.)
    u=float(np.max(np.asarray(prior['sigma_xy'])/size))
    residual=float(np.median(state['residuals'])) if state['residuals'] else None
    # Insufficient historical residual evidence stays explicitly uncertain.
    warm=min(1.,.65+.175*len(state['residuals']))
    stability=math.exp(-.5*((residual or 0.)/config['residual_scale'])**2)
    score=warm*math.exp(-gap/config['missing_tau_seconds'])*math.exp(-.5*(u/config['uncertainty_scale'])**2)*stability
    return dict(score=float(score),geometry_valid=True,missing_seconds=gap,
                uncertainty_ratio=u,residual=residual,reference_box=trusted_box,history_factor=warm)


def prompt_policy(state,prior,reliability,raw_identity,width,height,config=CONFIG):
    reasons=[];r=reliability['score']
    threshold=config['prompt_keep'] if state['prompt_enabled'] else config['prompt_on']
    if r<threshold:reasons.append('motion_unreliable')
    support=max(state['identity_confidence']*math.exp(-reliability['missing_seconds']/.5),raw_identity)
    explore=support<config['identity_accept']
    if explore and r<config['prompt_explore']:reasons.append('weak_identity_and_motion')
    ref=reliability['reference_box'];box=prior['box'];search=None
    if ref is None or box is None or prior['visible_fraction']<.02:
        reasons.append('invalid_or_outside_projection')
    else:
        refsize=np.maximum(np.asarray(ref[2:])-ref[:2],8.)
        size=np.maximum(np.asarray(box[2:])-box[:2],2.)
        ratio=size/refsize
        if np.any(ratio>2.) or np.any(ratio<.5):reasons.append('predicted_scale_unreliable')
        sigma=np.asarray(prior['sigma_xy'])
        padding=np.maximum(2*sigma,.15*size)+6
        if np.any(padding>config['padding_scale_limit']*refsize):reasons.append('uncertainty_exceeds_prompt_limit')
        search=clip_box(np.asarray(box)+np.r_[-padding,padding],width,height)
        area=float(np.maximum(search[2:]-search[:2],0).prod())
        if area/(width*height)>config['max_prompt_area']:reasons.append('prompt_too_large_in_image')
        if area/refsize.prod()>config['max_prompt_relative_area']:reasons.append('prompt_too_large_for_person')
        if np.any(search[2:]-search[:2]<4):reasons.append('empty_prompt')
    enabled=not reasons
    return dict(enabled=enabled,box=search if enabled else None,proposed_box=search,
                exploration=bool(explore),identity_support=float(support),reasons=reasons)


def appearance(desc,state):
    seed=appearance_similarity(desc,state['bank'][:1])
    adaptive=appearance_similarity(desc,state['bank'])
    return .60*seed+.40*adaptive,seed,adaptive


def dominant_fraction(mask):
    number,labels,stats,centroids=cv2.connectedComponentsWithStats(mask.astype(np.uint8),8)
    areas=stats[1:,cv2.CC_STAT_AREA]
    return float(areas.max()/max(1,areas.sum())) if len(areas) else 0.


def evaluate(candidate,identity,states,from_current,reliability,now,config=CONFIG):
    state=states[identity];reasons=[]
    if candidate['pixels']<200 or candidate['box'] is None:
        return dict(valid=False,reasons=['no_usable_mask'],score=0.,identity_score=0.,appearance=0.,
                    innovation=None,anchor_box=None,dominant_fraction=0.,identity_margin=0.)
    a,seed,adaptive=appearance(candidate['descriptor'],state)
    competitors=[dict(id=i,appearance=appearance(candidate['descriptor'],s)[0]) for i,s in states.items() if i!=identity]
    margin=a-max((c['appearance'] for c in competitors),default=0.)
    uniqueness=float(np.clip(.5+margin/.16,0,1))
    innovation=state['motion'].innovation(candidate['box'],from_current)
    anchor_box=transform_box(candidate['box'],from_current)
    temporal_available=now-state['last_trusted_time']<=.350001
    temporal=appearance_similarity(candidate['descriptor'],[state['last_descriptor']]) if temporal_available else 0.
    m=float(np.exp(-.125*min(innovation['mahalanobis2'],100))) if innovation is not None else 0.
    r=reliability['score'];wt=.15 if temporal_available else 0.
    identity_score=(.50*a+.15*uniqueness+wt*temporal+.20*r*m)/(.65+wt+.20*r)
    purity=dominant_fraction(candidate['small_mask'])
    score=.70*identity_score+.20*np.clip(candidate['iou_score'],0,1)+.10*candidate['presence']-.08*(1-purity)
    if candidate['iou_score']<config['min_iou']:reasons.append('low_mask_score')
    if candidate['presence']<config['min_presence']:reasons.append('low_object_presence')
    if identity_score<config['identity_candidate']:reasons.append('weak_identity_evidence')
    if purity<.25:reasons.append('highly_fragmented_mask')
    if score<config['min_match_score']:reasons.append('low_match_score')
    # No absolute position gate: unreliable prediction cannot veto a visual candidate.
    return dict(valid=not reasons,reasons=reasons,score=float(score),identity_score=float(identity_score),
        appearance=a,seed_appearance=seed,adaptive_appearance=adaptive,identity_margin=float(margin),
        competitors=competitors,uniqueness=uniqueness,temporal_available=temporal_available,
        temporal_similarity=float(temporal),motion_similarity=m,effective_motion_weight=.20*r,
        innovation=innovation,anchor_box=anchor_box,dominant_fraction=purity)


def pending_continuity(previous,current,config=CONFIG):
    if current['frame']-previous['frame']>config['max_pending_gap_frames']:return False
    if previous['anchor_box'] is None or current['anchor_box'] is None:return False
    a=box_to_z(previous['anchor_box']);b=box_to_z(current['anchor_box'])
    scale=np.maximum(np.exp(a[2:]),8.)
    distance=float(np.linalg.norm((b[:2]-a[:2])/scale))
    size=float(np.max(np.abs(a[2:]-b[2:])))
    sim=appearance_similarity(current['descriptor'],[previous['descriptor']])
    return distance<config['continuity_distance'] and size<config['continuity_log_scale'] and sim>.65


def adjudicate(state,candidate,evaluation,frame,now,reliability,exploration=False,config=CONFIG):
    if candidate is None:
        state['pending']=[]
        return dict(accepted=False,status='LOST',reason='no_eligible_candidate',pending_count=0,recovered=False)
    item=dict(frame=frame,time=now,anchor_box=evaluation['anchor_box'],descriptor=candidate['descriptor'])
    p=state['pending']
    if p and pending_continuity(p[-1],item,config):p.append(item)
    else:state['pending']=[item];p=state['pending']
    state['pending']=p[-config['long_recovery_frames']:];p=state['pending']
    gap=state['motion'].missed
    clear=(evaluation['identity_score']>=config['identity_accept'] and
           evaluation['identity_margin']>=config['identity_margin_accept'])
    continuous=(gap==0 and state['stable_streak']>0)
    if continuous and clear and not exploration:
        return dict(accepted=True,status='TRACKED',reason='continuous_identity_supported',pending_count=len(p),recovered=False)
    needed=config['long_recovery_frames'] if gap>=config['long_gap_frames'] else config['recovery_frames']
    if gap<config['long_gap_frames'] and evaluation['identity_score']>.92 and evaluation['identity_margin']>.10 and candidate['iou_score']>.85:
        needed=2
    if clear and len(p)>=needed:
        return dict(accepted=True,status='RECOVERED',reason='multi_frame_recovery_confirmed',pending_count=len(p),recovered=True)
    return dict(accepted=False,status='PENDING',reason='await_identity_confirmation',pending_count=len(p),needed=needed,recovered=False)


def update_trusted_motion(state,evaluation,frame,now,recovered,reliability):
    z=box_to_z(evaluation['anchor_box']) if evaluation['anchor_box'] is not None else None
    if z is None:return dict(updated=False,reset=False,source_frames=[],reason='observation_not_projectable')
    oldbox=state['last_anchor_box'];oldsize=np.maximum(np.asarray(oldbox[2:])-oldbox[:2],8.)
    delta=z[:2]-state['motion'].x[:2]
    residual=float(np.linalg.norm(delta/oldsize))
    reset=recovered or (reliability['score']<.25 and residual>1.)
    source_frames=[]
    if reset:
        old_frames=list(state['motion'].accepted_frames)
        motion=Motion.initial(evaluation['anchor_box'],now,frame)
        p=[v for v in state['pending'] if v['anchor_box'] is not None]
        if recovered and len(p)>=2:
            t=np.array([v['time'] for v in p]);values=np.array([box_to_z(v['anchor_box']) for v in p])
            dt=t-t.mean();velocity=dt@(values-values.mean(axis=0))/max(float(dt@dt),1e-8)
            motion.x[4:6]=np.clip(velocity[:2],-4*motion.seed_height,4*motion.seed_height)
            motion.x[6:]=np.clip(velocity[2:],-1.5,1.5)
            source_frames=[v['frame'] for v in p]
        motion.accepted_frames=old_frames+[frame];state['motion']=motion;state['residuals'].clear()
    else:
        state['motion'].update(z,frame);state['residuals'].append(min(residual,4.))
    state['last_anchor_box']=np.asarray(evaluation['anchor_box']);state['last_trusted_time']=now
    return dict(updated=True,reset=bool(reset),source_frames=source_frames,residual=residual)


def memory_policy(state,candidate,evaluation,recovered,config=CONFIG):
    reasons=[]
    if recovered:reasons.append('recovery_memory_hold')
    if state['memory_streak']<config['memory_streak']:reasons.append('insufficient_stable_memory_evidence')
    if evaluation['identity_score']<config['identity_memory']:reasons.append('identity_not_strong_for_memory')
    if evaluation['identity_margin']<config['identity_margin_memory']:reasons.append('identity_competition')
    if candidate['iou_score']<config['memory_quality']:reasons.append('mask_not_strong_for_memory')
    if candidate['presence']<config['memory_presence']:reasons.append('presence_not_strong_for_memory')
    if evaluation['dominant_fraction']<.70:reasons.append('fragmented_memory_candidate')
    return not reasons,reasons
