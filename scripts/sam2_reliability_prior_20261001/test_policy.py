"""Focused regression checks for the observed failure mechanisms."""
import copy,json
import numpy as np
from tracker_geometry import Motion,project,box_to_z,assign_candidates
from reliability import (CONFIG,new_state,motion_reliability,prompt_policy,evaluate,adjudicate,
                         update_trusted_motion,memory_policy)
from run import Camera,POSE,OUT

checks={};desc=dict(feature=np.array([1.,0.]),color=np.array([1.,0.]))
def seed():return new_state({'cond_frame_outputs':{0:{}},'non_cond_frame_outputs':{}},[400,700,520,1100],desc,0.,100)
def candidate(box=None,descriptor=desc):
    small=np.zeros((240,320),bool);small[80:130,80:100]=True
    return dict(pixels=8000,box=box or [400,700,520,1100],descriptor=descriptor,
                small_mask=small,iou_score=.95,presence=.99,source='memory')

s=seed();s['motion'].predict(.1);prior=s['motion'].image_prior(np.eye(3),2560,1920)
r=motion_reliability(s,prior,np.eye(3),.1)
assert prompt_policy(s,prior,r,.95,2560,1920)['enabled']
checks['stable_initial_track_can_use_bounded_prompt']=True
for k in range(2,32):s['motion'].missed+=1;s['motion'].predict(k/10)
prior=s['motion'].image_prior(np.eye(3),2560,1920);lost=motion_reliability(s,prior,np.eye(3),3.1)
assert lost['score']<r['score']*.1 and not prompt_policy(s,prior,lost,.95,2560,1920)['enabled']
checks['lost_uncertain_prediction_disables_prompt']=True

before=lost['score'];prior['box']=np.array([0,0,2560,1920.])
assert motion_reliability(s,prior,np.eye(3),3.1)['score']==before
checks['inflating_predicted_box_cannot_raise_reliability']=True

s=seed();s['motion'].predict(.1)
c=candidate([1900,700,2020,1100]);ev=evaluate(c,'A',{'A':s},np.eye(3),{'score':.01},.1)
assert ev['valid'] and ev['innovation']['mahalanobis2']>25
assert adjudicate(s,c,ev,101,.1,{'score':.01})['accepted']
checks['unreliable_motion_does_not_veto_strong_visual_candidate']=True

s=seed();s['motion'].predict(.1)
weak=dict(feature=np.array([.65,np.sqrt(1-.65**2)]),color=np.array([.65,np.sqrt(1-.65**2)]))
c=candidate(descriptor=weak)
low=evaluate(c,'A',{'A':s},np.eye(3),{'score':0.},.1)
high=evaluate(c,'A',{'A':s},np.eye(3),{'score':.9},.1)
assert high['identity_score']>CONFIG['identity_accept']>low['identity_score']
checks['reliable_motion_can_support_weaker_appearance']=True

s=seed();s['motion'].missed=2;s['stable_streak']=0
main=copy.deepcopy(s['motion']);mem=copy.deepcopy(s['memory'])
for k in range(1,4):
    c=candidate([400+5*k,700,520+5*k,1100])
    ev=evaluate(c,'A',{'A':s},np.eye(3),{'score':.1},k/10)
    ev['identity_score']=.84;ev['identity_margin']=.07
    decision=adjudicate(s,c,ev,100+k,k/10,{'score':.1})
    if k<3:assert not decision['accepted']
    assert np.array_equal(s['motion'].x,main.x) and np.array_equal(s['motion'].P,main.P)
    assert s['motion'].last_observation_frame==100 and s['memory']==mem
assert decision['accepted'] and decision['recovered']
checks['pending_sequence_isolated_from_primary_state_and_memory']=True
result=update_trusted_motion(s,ev,103,.3,True,{'score':.1})
assert result['reset'] and result['source_frames']==[101,102,103]
assert abs(s['motion'].x[4]-50)<1e-6
checks['confirmed_recovery_rebuilds_motion_from_observed_sequence']=True
s['memory_streak']=10
write,reasons=memory_policy(s,c,ev,True)
assert not write and 'recovery_memory_hold' in reasons
checks['new_recovery_cannot_immediately_write_memory']=True

s=seed();s['motion'].missed=20;s['stable_streak']=0
for k in range(1,3):
    c=candidate([400+5*k,700,520+5*k,1100]);ev=evaluate(c,'A',{'A':s},np.eye(3),{'score':0.},k/10)
    adjudicate(s,c,ev,100+k,k/10,{'score':0.})
c=candidate([1800,200,2000,600]);ev=evaluate(c,'A',{'A':s},np.eye(3),{'score':0.},.3)
d=adjudicate(s,c,ev,103,.3,{'score':0.})
assert d['pending_count']==1 and not d['accepted']
checks['discontinuous_recovery_candidates_restart_confirmation']=True

cs=[dict(owner='A'),dict(owner='A'),dict(owner='B')];es=[dict(valid=True,score=x) for x in [.91,.90,.89]]
assign,_=assign_candidates(['A','B'],cs,es,[[0,2],[1]],CONFIG)
assert assign['A']['candidate']==1 and assign['B']['candidate']==2
checks['joint_assignment_preserves_unique_candidate_groups']=True

K=np.array([[1100.,0,1280],[0,1100.,960],[0,0,1.]])
a=.1;R=np.array([[np.cos(a),0,np.sin(a)],[0,1,0],[-np.sin(a),0,np.cos(a)]])
H=K@R.T@np.linalg.inv(K);p=np.array([[1280,960],[800,700.]])
q,ok=project(p,H);back,ok2=project(q,np.linalg.inv(H))
assert ok.all() and ok2.all() and np.max(abs(back-p))<1e-8 and q[0,0]<1280
checks['rotation_direction_and_round_trip']=True
camera=Camera(POSE)
for f in range(100,121):camera.set_current_frame(f)
before=camera.matrices(120,True);camera.R[121:]=np.nan
assert all(np.array_equal(a,b) for a,b in zip(before,camera.matrices(120,True)))
try:camera.matrices(121,True);raise RuntimeError('future access accepted')
except AssertionError:pass
checks['future_pose_access_forbidden_and_suffix_invariant']=True
OUT.mkdir(parents=True,exist_ok=True)
result=dict(status='passed',checks=checks)
(OUT/'unit_checks.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
