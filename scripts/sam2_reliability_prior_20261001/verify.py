"""Validate frozen inputs, time ordering, pending isolation and actual prompt use."""
import gzip,json
from collections import Counter
from pathlib import Path
import cv2
import numpy as np
from run import ROOT,OUT,OLD,BRANCH,decode,digest,read_json,write_json

protocol=read_json(OUT/'protocol.json');assert read_json(OUT/'unit_checks.json')['status']=='passed'
for item in protocol['source_hashes'].values():assert digest(ROOT/item['path'])==item['sha256'],item['path']
for name,checksum in protocol['code_hashes'].items():assert digest(Path(__file__).parent/name)==checksum,name
with gzip.open(OUT/'decisions.jsonl.gz','rt') as h:rows={r['frame']:r for r in map(json.loads,h)}
assert sorted(rows)==list(range(100,340))
with gzip.open(OUT/'candidates_rle.jsonl.gz','rt') as h:candidates={(r['frame'],r['candidate_id']):r['rle'] for r in map(json.loads,h)}
assert set(candidates)=={(f,c['candidate_id']) for f,row in rows.items() for c in row['branches'][BRANCH]['candidates']}
initial=read_json(OLD/'initial_masks.json');records=set();coverage=Counter()
with gzip.open(OUT/'masks_rle.jsonl.gz','rt') as h:
    for r in map(json.loads,h):
        key=(r['frame'],r['id']);assert key not in records;records.add(key)
        d=rows[r['frame']]['branches'][BRANCH]['decisions'][r['id']];selected=d['selected_candidate']
        m=decode(r['rle']);assert m.shape==(1920,2560);coverage[r['id']]+=int(m.any())
        if selected=='initial':assert r['rle']==initial[r['id']]
        elif selected is None:assert not m.any()
        else:assert r['rle']==candidates[(r['frame'],selected)]
assert records=={(f,i) for f in range(100,340) for i in 'ABCDE'}
isolated=0;holds=0;disabled=0;enabled=0;recoveries=0
for f,row in rows.items():
    assert row['current_camera_pose_frame']==f
    block=row['branches'][BRANCH];groups=[]
    for i,d in block['decisions'].items():
        mem=d['memory_after'];assert mem['references'][0]==100
        assert len(mem['references'])<=3 and len(mem['recent'])<=7 and max(mem['references']+mem['recent'])<=f
        if f==100:continue
        prior=d['prior'];assert prior['last_observation_frame']<f
        covariance=np.asarray(d['posterior_covariance']);assert np.isfinite(covariance).all()
        assert np.linalg.eigvalsh(covariance).min()>-1e-7
        if not d['memory_written']:
            holds+=1;assert d['memory_before']==mem and d['appearance_bank_before']==d['appearance_bank_count']
        else:assert d['selected_candidate'] and d['status']=='TRACKED'
        if d['status'] in ['PENDING','LOST']:
            isolated+=1;assert not d['motion_updated'] and not d['memory_written'] and d['selected_candidate'] is None
            assert np.array_equal(d['posterior_state'],prior['anchor_state'])
            assert np.array_equal(covariance,prior['anchor_covariance'])
            assert d['last_observation_frame_after']==prior['last_observation_frame']
        if d['status']=='RECOVERED':
            recoveries+=1;assert not d['memory_written'] and len(d['recovery_evidence_frames'])>=2
            assert max(d['recovery_evidence_frames'])==f
        if d['motion_updated']:assert d['last_observation_frame_after']==f
        guided=[c for c in block['candidates'] if c['owner']==i and c['source']=='motion_box']
        policy=d['prompt_policy']
        if not policy['enabled']:
            disabled+=1;assert not guided and policy['box'] is None and prior['search_box'] is None
        else:
            enabled+=1;assert len(guided)==1 and np.allclose(policy['box'],guided[0]['prompt_box'])
            box=np.array(policy['box']);ref=np.array(d['motion_reliability']['reference_box'])
            area=np.prod(box[2:]-box[:2]);refarea=np.prod(np.maximum(ref[2:]-ref[:2],8.))
            assert area/(2560*1920)<=protocol['config']['max_prompt_area']+1e-9
            assert area/refarea<=protocol['config']['max_prompt_relative_area']+1e-9
        if d['assigned_group'] is not None:groups.append(d['assigned_group'])
    assert len(groups)==len(set(groups))
videos={}
for name,shape in [('tracking_reliability.mp4',(1088,1700)),('three_versions.mp4',(648,1920))]:
    cap=cv2.VideoCapture(str(OUT/'presentation'/name));assert abs(cap.get(cv2.CAP_PROP_FPS)-10)<.01;n=0
    while True:
        ok,im=cap.read()
        if not ok:break
        assert im.shape[:2]==shape;n+=1
    cap.release();assert n==240;videos[name]=dict(frames=n,width=shape[1],height=shape[0],fps=10)
result=dict(status='passed',frames=240,final_mask_records=len(records),candidate_mask_records=len(candidates),
    source_and_prior_results_unchanged=True,frozen_inference_code_unchanged=True,
    no_future_pose_or_observation_indices=True,initial_masks_identical=True,
    pending_or_lost_primary_states_isolated=isolated,memory_holds_verified=holds,
    disabled_prompts_absent=disabled,enabled_prompts_within_limits=enabled,
    recovered_frames_with_memory_held=recoveries,one_to_one_frames_verified=239,
    nonempty_masks=dict(coverage),videos=videos,interpretation='Mechanical integrity and policy checks, not identity ground truth.')
write_json(OUT/'verification.json',result);print(json.dumps(result,indent=2))
