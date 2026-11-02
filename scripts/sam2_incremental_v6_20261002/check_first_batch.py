from pathlib import Path
import gzip,json,hashlib
ROOT=Path(__file__).resolve().parents[2]
def rows(p):
    with gzip.open(p,'rt') as f:return [json.loads(s) for s in f]
old=ROOT/'outputs/sam2_g1_v6_v8_20261001/04/v6'
new=ROOT/'outputs/sam2_incremental_v6_20261002/04/batch_01'
a={(r['frame'],r['id']):r['rle'] for r in rows(old/'masks_rle.jsonl.gz')}
b={(r['frame'],r['id']):r['rle'] for r in rows(new/'masks_rle.jsonl.gz')}
same=sum(a[k]==v for k,v in b.items());assert a==b,(same,len(a),len(b))
source=ROOT/'scripts/sam2_motion_tracker_v1_20261001/core.py';used=ROOT/'scripts/sam2_g1_v6_v8_20261001/core.py'
assert source.read_bytes()==used.read_bytes()
result=dict(status='passed',check='With same three seeds and no earlier donors, all masks exactly equal old V6',identical_mask_rows=same,core_sha256=hashlib.sha256(used.read_bytes()).hexdigest())
(new.parent.parent/'frozen_v6_check.json').write_text(json.dumps(result,indent=2));print(result)
