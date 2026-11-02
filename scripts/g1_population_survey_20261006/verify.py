"""Independent descriptor replay and response-accounting checks."""
from pathlib import Path
from collections import Counter
import json,hashlib,numpy as np
from analyze import features
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/g1_population_survey_20261006'
def read(p):return json.loads(p.read_text('utf8'))
bank=ROOT/'outputs/transformer_matched97_20261006/new';meta=read(bank/'data/samples.json');data=np.load(bank/'data/dataset.npz')
inv={r['key']:r for r in read(ROOT/'outputs/g1_factor_inventory_20261006/existing_windows.json')}
calcache={};error={k:0. for k in 'DEG'}
for i,m in enumerate(meta):
    cid=m['clip'];n=int(cid)
    if cid not in calcache:
        version='sam2_g1_v6_v8_20261001' if n<=10 else 'sam2_g1_expansion_20261005' if n<=21 else 'sam2_g1_expansion_v2_20261005' if n<=31 else 'sam2_g1_matched97_20261006'
        calcache[cid]=dict(np.load(ROOT/'outputs'/version/cid/'input/reference.npz'))
    src=calcache[cid];idx=m['frames']
    cal=dict(R=src['R'][idx],K=src['K'],times_ns=src['times_ns'][idx],position=np.zeros((21,3)))
    value=features(data['boxes'][i],cal)
    for k in error:error[k]=max(error[k],abs(value[k]-inv[m['key']]['features'][k]))
# Existing stored H matrices are float32; rebuilding from R/K changes G by
# sub-millipixel/second rounding. Bound G separately in image pixels/second.
assert error['D']<1e-10 and error['E']<1e-10 and error['G']*1920<.001,error
ledger=read(OUT/'response_ledger.json');measured=read(OUT/'measured_features.json');result=read(OUT/'distribution.json');sample=read(OUT/'sample_locked.json')
assert len(ledger)==2507 and len(measured)==197
assert {r['key'] for r in measured}=={r['key'] for r in ledger if r['status']=='measured_G1'}
eligible=Counter(r['status'] for r in ledger if r['automatic_eligible'])
assert dict(eligible)==dict(measured_G1=197,visual_target_exclusion=70,outside_visual_domain=80,mask_measurement_failure=2),eligible
assert all(np.isclose(r['survey_weight']*r['inclusion_probability'],1) for r in measured)
assert result['groups']['train_source']['n']==295
assert np.isclose(sum(r['share'] for r in result['groups']['survey_all']['cells']),1)
assert result['audit']['visual_review_sha256']==hashlib.sha256((OUT/'visual_review.json').read_bytes()).hexdigest()
audit=dict(prior296_descriptor_replay_max_abs_error=error,G_max_pixel_per_second_difference=error['G']*1920,
    G_tolerance_reason='Recomputed R/K homography versus existing float32 stored H; tolerance 0.001 pixel/second.',automatic_candidate_accounting=dict(eligible),
    accepted_ids_match_ledger=True,positive_inverse_inclusion_weights=True,training_union295=True,
    joint_probabilities_sum_to_one=True,visual_review_hash_matches=True)
(OUT/'verification.json').write_text(json.dumps(audit,indent=2),'utf8')
print(json.dumps(audit,indent=2))
