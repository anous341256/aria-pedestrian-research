"""Check video completeness, mask records, and preservation of existing A/B behavior."""
from collections import Counter
import gzip
import json
import cv2
import numpy as np

from five_people import OUT, PREVIOUS, VIDEO, ROOT, W, H, decode
from common import read_json, write_json, digest

protocol=read_json(OUT/'protocol.json')
records=read_json(OUT/'records.json')
assert len(records)==240
assert digest(VIDEO)==protocol['source_sha256']
assert digest(PREVIOUS/'protocol.json')==protocol['previous_protocol_sha256']
by_frame={row['frame']:row for row in records}
previous={}
with gzip.open(PREVIOUS/'night_identity/masks_rle.jsonl.gz','rt') as handle:
    for line in handle:
        row=json.loads(line)
        if row['variant'] in ('baseline','negative_points'):
            previous[(row['frame'],row['variant'])]=row['rle']
donor={}
with gzip.open(ROOT/'outputs/sam2_fullvideos_20260927/201129/sam2_raw_v1/masks_rle.jsonl.gz','rt') as handle:
    for line in handle:
        row=json.loads(line)
        if row['id']==2 and 100<=row['frame']<340:
            donor[row['frame']]=row['rle']
counts=Counter()
reproduction=Counter()
all_ids={'A','B','C','D','E'}
with gzip.open(OUT/'masks_rle.jsonl.gz','rt') as handle:
    seen=set()
    for line in handle:
        row=json.loads(line)
        f,variant,identity=row['frame'],row['variant'],row['id']
        key=(f,variant,identity)
        assert key not in seen
        seen.add(key)
        assert identity in all_ids and variant in protocol['variants'] and 100<=f<340
        mask=decode(row['rle'])
        assert mask.shape==(H,W)
        assert int(mask.sum())==by_frame[f]['variants'][variant][identity]['pixels']
        counts[(variant,identity)]+=1
        reference=None
        label=None
        if identity=='A' and variant=='baseline':
            reference=donor.get(f)
            label='A_archived'
        elif identity=='B':
            source_variant='baseline' if variant=='baseline' else 'negative_points'
            reference=previous[(f,source_variant)]
            label='B_'+variant
        if reference is not None:
            reproduction[label+'_checked']+=1
            reproduction[label+'_identical']+=int(np.array_equal(mask,decode(reference)))
assert dict(counts)=={(v,i):240 for v in protocol['variants'] for i in all_ids}
video_checks={}
for name,shape in [('five_people.mp4',(1132,1280)),('comparison.mp4',(892,1920))]:
    cap=cv2.VideoCapture(str(OUT/name))
    assert abs(cap.get(cv2.CAP_PROP_FPS)-10)<.01
    total=0
    while True:
        ok,frame=cap.read()
        if not ok:break
        assert frame.shape[:2]==shape
        total+=1
    cap.release()
    assert total==240
    video_checks[name]=dict(decoded_frames=total,width=shape[1],height=shape[0],fps=10)
audit=read_json(OUT/'applied_negative_prompts.json')
assert len(audit)==12
assert {(r['frame'],r['target']) for r in audit}=={(f,i) for f in [160,170,190] for i in 'BCDE'}
assert all(r['labels']==[0]*4 and r['passed_to_prompt_encoder'] for r in audit)
diagnostics={}
for variant in protocol['variants']:
    diagnostics[variant]=dict(
        nonempty_frames={i:sum(r['variants'][variant][i]['pixels']>0 for r in records) for i in 'ABCDE'},
        all_five_nonempty_frames=sum(all(r['variants'][variant][i]['pixels']>0 for i in 'ABCDE') for r in records),
        frames_with_pair_overlap_over_100_pixels={pair:sum(r['variants'][variant]['pairwise_overlap_pixels'][pair]>100
                                                             for r in records)
                                                  for pair in ['AB','AC','AD','AE','BC','BD','BE','CD','CE','DE']})
result=dict(status='passed',videos=video_checks,mask_records=len(seen),source_unchanged=True,
            previous_protocol_unchanged=True,old_identity_reproduction=dict(reproduction),
            identical_branches_before_first_intervention=all(all(r['branches_equal'].values()) for r in records if r['frame']<160),
            negative_prompt_events=len(audit),diagnostics=diagnostics,
            interpretation='Nonempty and overlap counts describe model outputs, not identity accuracy or segmentation quality.')
write_json(OUT/'verification.json',result)
print(json.dumps(result,indent=2),flush=True)
