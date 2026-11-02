from pathlib import Path
import json, hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/"outputs/rgb_conditioned_forecast_20261020"
SOURCE=ROOT/"outputs/translation_forecast_20261020"
ARMS=["no_rgb","local_rgb","scene_rgb","both_rgb"]
def read(p):return json.loads(Path(p).read_text(encoding="utf-8-sig"))
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+".partial")
    tmp.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False),encoding="utf-8");tmp.replace(p)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
    return h.hexdigest()
def source_video(m):return ROOT/f'outputs/sam2_g1_v6_v8_20261001/{m["clip"]}/input/rgb.mp4'
def numeric(d):
    xy=d["xy"][:,:16]/2560.
    wh=(d["boxes"][:,:16,2:]-d["boxes"][:,:16,:2])/2560.
    delta=np.zeros_like(xy);delta[:,1:]=np.diff(xy,axis=1)
    rel=xy-xy[:,-1,None]
    person=np.concatenate([xy,rel,delta,wh,d["seconds"][:,:16,None]/1.5],-1)
    camera=np.concatenate([d["cam"][:,:16],d["translation"][:,:16]],-1)
    return person.astype(np.float32),camera.astype(np.float32)
