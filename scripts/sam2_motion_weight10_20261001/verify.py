"""Verify temporal, identity, memory, mask, and artifact consistency after inference."""
from collections import Counter
import gzip
import json
from pathlib import Path
import cv2
import numpy as np
from run import ROOT,OUT,OLD,BRANCHES,decode
from common import read_json,write_json,digest

protocol=read_json(OUT/'protocol.json')
assert read_json(OUT/'unit_checks.json')['status']=='passed'
for item in protocol['source_hashes'].values():
    assert digest(ROOT/item['path'])==item['sha256'],item
for name,checksum in protocol['code_hashes'].items():
    assert digest(Path(__file__).parent/name)==checksum,name
rows={}
with gzip.open(OUT/'decisions.jsonl.gz','rt') as handle:
    for line in handle:
        row=json.loads(line)
        assert row['frame'] not in rows
        rows[row['frame']]=row
assert sorted(rows)==list(range(100,340))
initial=read_json(OLD/'initial_masks.json')
candidate_rles={}
with gzip.open(OUT/'candidates_rle.jsonl.gz','rt') as handle:
    for line in handle:
        row=json.loads(line)
        key=(row['frame'],row['branch'],row['candidate_id'])
        assert key not in candidate_rles
        candidate_rles[key]=row['rle']
expected_candidates={(f,b,c['candidate_id']) for f,row in rows.items() for b in BRANCHES
                     for c in row['branches'][b]['candidates']}
assert set(candidate_rles)==expected_candidates
counts=Counter();nonempty={b:Counter() for b in BRANCHES};pixels={}
with gzip.open(OUT/'masks_rle.jsonl.gz','rt') as handle:
    for line in handle:
        row=json.loads(line);f,b,i=row['frame'],row['branch'],row['id']
        key=(f,b,i);assert key not in pixels
        mask=decode(row['rle']);assert mask.shape==(1920,2560)
        area=int(mask.sum());pixels[key]=area
        counts[(b,i)]+=1;nonempty[b][i]+=int(area>0)
        decision=rows[f]['branches'][b]['decisions'][i]
        selected=decision['selected_candidate']
        if selected=='initial':assert row['rle']==initial[i]
        elif selected is None:assert area==0
        else:
            assert row['rle']==candidate_rles[(f,b,selected)]
            candidate=next(c for c in rows[f]['branches'][b]['candidates'] if c['candidate_id']==selected)
            assert candidate['valid'] and area==candidate['pixels']
assert dict(counts)=={(b,i):240 for b in BRANCHES for i in 'ABCDE'}
statuses={b:Counter() for b in BRANCHES};rejections={b:Counter() for b in BRANCHES}
memory_holds=0;one_to_one_frames=0
for f,row in rows.items():
    assert row['current_camera_pose_frame']==f
    for branch in BRANCHES:
        block=row['branches'][branch]
        selected_groups=[]
        for identity,decision in block['decisions'].items():
            statuses[branch][decision['status']]+=1
            after=decision['memory_after']
            assert after['references'][0]==100 and len(after['references'])<=3 and len(after['recent'])<=7
            assert max(after['references']+after['recent'])<=f
            if f==100:continue
            prior=decision['prior'];assert prior['last_observation_frame']<f
            assert np.isfinite(prior['anchor_state']).all()
            covariance=np.array(decision['posterior_covariance'])
            assert np.isfinite(covariance).all() and np.linalg.eigvalsh(covariance).min()>-1e-7
            if not decision['memory_written']:
                assert decision['memory_before']==after;memory_holds+=1
            else:
                assert decision['selected_candidate'] is not None and decision['status']=='TRACKED'
                assert f in after['recent'] or f in after['references']
            if decision['selected_candidate'] is not None:selected_groups.append(decision['selected_group'])
            else:
                assert decision['missed_frames_after']==prior['missed_frames']+1
                assert pixels[(f,branch,identity)]==0
        assert len(selected_groups)==len(set(selected_groups));one_to_one_frames+=1
        for candidate in block['candidates']:
            for reason in candidate['reasons']:rejections[branch][reason]+=1
videos={}
for name,shape in [('tracking_diagnostics.mp4',(1088,1700)),('three_way_comparison.mp4',(648,1920))]:
    cap=cv2.VideoCapture(str(OUT/'presentation'/name));assert abs(cap.get(cv2.CAP_PROP_FPS)-10)<.01
    n=0
    while True:
        ok,image=cap.read()
        if not ok:break
        assert image.shape[:2]==shape;n+=1
    cap.release();assert n==240
    videos[name]=dict(frames=n,width=shape[1],height=shape[0],fps=10)
result=dict(status='passed',frames=240,object_mask_records=len(pixels),candidate_mask_records=len(candidate_rles),
            source_hashes_unchanged=True,frozen_inference_code_unchanged=True,
            initial_masks_identical=True,all_selected_outputs_equal_saved_visual_candidates=True,
            prior_history_strictly_before_current_frame=True,no_future_camera_indices=True,
            rejected_memory_holds_verified=memory_holds,one_to_one_frames_verified=one_to_one_frames,
            covariance_checks_passed=True,videos=videos,
            diagnostic_counts=dict(statuses={b:dict(v) for b,v in statuses.items()},
                                   nonempty_masks={b:dict(v) for b,v in nonempty.items()},
                                   candidate_rejection_reasons={b:dict(v) for b,v in rejections.items()}),
            interpretation='Integrity and rule consistency checks. These do not establish correct identities or mask accuracy.')
write_json(OUT/'verification.json',result)
print(json.dumps(result,indent=2),flush=True)
