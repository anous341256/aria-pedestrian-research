import sys,json,gzip,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/sam2_g1_v6_v8_20261001'
sys.path.insert(0,str(ROOT/'scripts/midterm_20260927'))
from common import read_json,write_json,digest
rows=[];totals={v:dict(post_seed_nonempty=0,denominator=0,main_nonempty=0,main_denominator=0,memory_writes=0,prompts=0,seconds=0) for v in ['v6','v8']}
for k in range(1,11):
    cid=f'{k:02d}';folder=OUT/cid;s=read_json(folder/'comparison/stats.json');meta=read_json(folder/'input/metadata.json');n=s['frames'];people=s['people']
    row=dict(clip=cid,source_id=meta['source']['id'],start_s=meta['source']['start_s'],end_s=meta['source']['end_s'],frames=n,people=people,pose_held_frames=meta['pose_held_frames'],methods={})
    for v in totals:
        ss=s['branches'][v]['ids'];r=read_json(folder/v/'runtime.json')
        values=dict(post_seed_nonempty=sum(i['post_seed_nonempty'] for i in ss.values()),denominator=(n-1)*people,main_nonempty=ss['A']['post_seed_nonempty'],main_denominator=n-1,
                    memory_writes=sum(i['memory_writes'] for i in ss.values()),prompts=sum(i['prompts'] for i in ss.values()),seconds=r['elapsed_seconds'])
        row['methods'][v]=values
        for key,value in values.items():totals[v][key]+=value
    rows.append(row)
write_json(OUT/'summary.json',dict(clips=rows,totals=totals,
    scope='10 selected intervals, 530 RGB frames, 29 person-track instances. Seed frames excluded from coverage. Six recording groups; clips are not independent identities.',
    interpretation='Output coverage measures nonempty masks only, NOT tracking accuracy or recall against GT. Empty outputs can reflect occlusion, image exit, rejection or failure.',
    checkpoint=dict(repo='facebook/sam2.1-hiera-large',revision='665f8e2ad61cf5f53d65644ff27c8ee525124610',sha256='2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318'),
    model_runtime_note='Both branches use identical cached weights and adapter. Optional SAM2 CUDA hole-filling extension unavailable in both; no change during this trial.',
    sampling='Initial frame plus quarter, half, three-quarter and final frame of every clip (50 timestamps). Targeted failure checks added separately; no exhaustive pixel GT.'))
for r in rows:
    print(r['clip'],r['people'], ' | '.join(f"{v}: all {m['post_seed_nonempty']}/{m['denominator']}, A {m['main_nonempty']}/{m['main_denominator']}" for v,m in r['methods'].items()))
print('TOTAL',json.dumps(totals))
