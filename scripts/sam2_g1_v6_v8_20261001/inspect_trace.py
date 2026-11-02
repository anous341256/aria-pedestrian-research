import sys,json,gzip
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/sam2_g1_v6_v8_20261001'
cid,ident=sys.argv[1:3];frames=list(map(int,sys.argv[3:]))
for v in ['v6','v8']:
    with gzip.open(OUT/cid/v/'decisions.jsonl.gz','rt') as f:rows=[json.loads(l) for l in f]
    for fr in frames:
        row=rows[fr]['branches']['motion_with_camera'];d=row['decisions'][ident]
        keys=['status','reason','memory_written','memory_reason','identity_score','identity_margin','selected_candidate','tentative_candidate','motion_reliability','prompt_policy','pending_count']
        print(v,fr,ident,json.dumps({k:d.get(k) for k in keys},ensure_ascii=False))
        keys=['candidate_id','box','iou_score','presence','score','identity_score','identity_margin','valid','reasons','appearance','seed_appearance','competition','continuity_score','dominant_fraction']
        for c in row['candidates']:
            if c['owner']==ident:print(json.dumps({k:c.get(k) for k in keys},ensure_ascii=False))
