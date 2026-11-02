"""Run each branch in an isolated process; completed outputs are never overwritten."""
import os,sys,json,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/sam2_g1_v6_v8_20261001';HERE=Path(__file__).parent
for k in range(1,11):
    cid=f'{k:02d}';p=OUT/cid/'input/protocol.json';data=json.loads(p.read_text())
    assert data['review_status']=='first_frame_identity_and_mask_extent_reviewed'
    for version in ['v6','v8']:
        dest=OUT/cid/version
        if (dest/'runtime.json').exists():continue
        env=os.environ.copy();env['G1_CLIP']=cid
        log=OUT/f'{cid}_{version}.log'
        with log.open('w') as f:
            result=subprocess.run([sys.executable,str(HERE/(version+'_run.py'))],env=env,stdout=f,stderr=subprocess.STDOUT)
        if result.returncode:
            print(log.read_text(),flush=True);raise RuntimeError(f'{cid} {version} failed')
        runtime=json.loads((dest/'runtime.json').read_text())
        print(cid,version,round(runtime['elapsed_seconds'],1),runtime['summary'],flush=True)
print('ALL 20 RUNS COMPLETE',flush=True)
