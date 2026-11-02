import time
from pathlib import Path
from render import OUT,render_clip,montage
for k in range(1,11):
    cid=f'{k:02d}';begin=time.monotonic()
    while not (OUT/cid/'v8/runtime.json').exists():
        if time.monotonic()-begin>1200:raise RuntimeError('Inference did not complete within 20 minutes')
        time.sleep(2)
    render_clip(cid)
montage()
