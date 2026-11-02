"""Focused checks for geometry, causality, missing observations and identity conflicts."""
import copy
import json
from pathlib import Path
import tempfile
import numpy as np
from core import (Motion,DEFAULTS,project,assign_candidates,memory_decision,evaluate_candidate,
                  cluster_candidates,clean)
from run import Camera,POSE,OUT

checks={}
K=np.array([[1100.,0,1280],[0,1100.,960],[0,0,1.]])
angle=.10
rotation=np.array([[np.cos(angle),0,np.sin(angle)],[0,1,0],[-np.sin(angle),0,np.cos(angle)]])
H=K@rotation.T@np.linalg.inv(K)
points=np.array([[1280,960],[800,700],[1600,1400.]])
warped,valid=project(points,H)
back,valid_back=project(warped,np.linalg.inv(H))
assert valid.all() and valid_back.all()
assert np.max(np.abs(back-points))<1e-8
assert warped[0,0]<1280
checks['rotation_round_trip_and_direction']=True

motion=Motion.initial([1200,700,1360,1220],0,100)
motion.predict(.1)
prior=motion.image_prior(H,2560,1920)
center=(prior['box'][:2]+prior['box'][2:])/2
assert np.linalg.norm(center-warped[0])<1e-8
checks['stationary_world_target_moves_with_camera']=True

motion=Motion.initial([400,700,520,1100],0,100)
errors=[]
for i in range(1,31):
    motion.predict(i/10)
    actual=np.array([400,700,520,1100.])+np.array([8,2,8,2])*i
    innovation=motion.innovation(actual,np.eye(3))
    if i>20:errors.append(float(np.linalg.norm(innovation['delta'][:2])))
    motion.update(innovation['z'],100+i)
assert max(errors)<4.,errors
checks['constant_velocity_prediction_converges']=True

last=motion.last_observation_frame
old_uncertainty=float(np.trace(motion.P[:2,:2]))
for i in range(31,41):
    motion.predict(i/10);motion.missed+=1
assert motion.last_observation_frame==last
assert np.trace(motion.P[:2,:2])>old_uncertainty
assert np.linalg.eigvalsh(motion.P).min()>0
checks['missing_observations_increase_uncertainty_without_fabricating_history']=True

candidates=[dict(owner='A'),dict(owner='A'),dict(owner='B')]
evaluations=[dict(valid=True,score=.91),dict(valid=True,score=.90),dict(valid=True,score=.89)]
assigned,_=assign_candidates(['A','B'],candidates,evaluations,[[0,2],[1]],DEFAULTS)
assert assigned['A']['candidate']==1 and assigned['B']['candidate']==2
checks['joint_assignment_uses_alternative_to_avoid_duplicate_identity']=True

image=np.zeros((24,32),bool);image[6:20,8:16]=True
groups,_=cluster_candidates([dict(candidate_id='A',pixels=2000,small_mask=image),
                            dict(candidate_id='B',pixels=2000,small_mask=image.copy())],DEFAULTS)
assert len(groups)==1
checks['duplicate_masks_share_one_assignment_group']=True

descriptor=dict(feature=np.array([1.,0.]),color=np.array([1.,0.]))
bad=dict(pixels=1000,box=[2000,100,2300,600],descriptor=descriptor,iou_score=.99,presence=.99)
seed=Motion.initial([400,700,520,1100],0,100);seed.predict(.1)
evaluation=evaluate_candidate(bad,seed,np.eye(3),[descriptor],DEFAULTS,2560,1920)
assert not evaluation['valid'] and 'motion_gate' in evaluation['reasons']
write,reasons=memory_decision(evaluation,1.,DEFAULTS)
assert not write
checks['strong_visual_score_does_not_override_large_motion_conflict']=True

camera=Camera(POSE)
for f in range(100,121):camera.set_current_frame(f)
before=camera.matrices(120,True)
camera.R[121:]=np.nan
after=camera.matrices(120,True)
assert all(np.array_equal(a,b) for a,b in zip(before,after))
try:
    camera.matrices(121,True)
    raise RuntimeError('Future access was accepted')
except AssertionError:pass
assert all(np.array_equal(v,np.eye(3)) for v in camera.matrices(120,False))
checks['future_pose_suffix_cannot_affect_current_frame']=True
checks['future_pose_access_rejected']=True
checks['disabled_camera_is_identity']=True

raw=np.load(POSE)
r=raw['R'][100:340]
assert np.isfinite(r).all()
turn=max(float(np.degrees(np.arccos(np.clip((np.trace(r[0].T@q)-1)/2,-1,1)))) for q in r)
checks['all_240_camera_rotations_valid']=True
OUT.mkdir(parents=True,exist_ok=True)
result=dict(status='passed',checks=checks,max_constant_velocity_error_px=max(errors),
            maximum_rotation_from_start_degrees=turn)
(OUT/'unit_checks.json').write_text(json.dumps(clean(result),indent=2)+'\n')
print(json.dumps(clean(result),indent=2))
