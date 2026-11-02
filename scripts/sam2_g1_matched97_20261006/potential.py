from pathlib import Path
import json,numpy as np
from window_motion import *
OUT=Path(__file__).resolve().parents[2]/'outputs/sam2_g1_matched97_20261006'
for p in sorted(OUT.glob('*/census.json')):
 cid=p.parent.name;a=np.load(p.parent/'input/reference.npz');cs=json.loads(p.read_text('utf8'));out=[]
 for e in cs['segments']:
  rows={r['frame']:r for r in e['rows']};fs=sorted(rows);n=0
  for f in range(fs[0],fs[-1]-19,5):
   boxes=np.array([rows[i]['box'] for i in range(f,f+21)])
   if (boxes[:,0]<=5).any() or (boxes[:,1]<=5).any() or (boxes[:,2]>=2555).any() or (boxes[:,3]>=1915).any():continue
   if eligible(rotation_stats(a['R'][f:f+21],a['times_ns'][f:f+21]/1e9)):n+=1
  if n:out.append((e['id'],n))
 print(cid,sum(n for _,n in out),out,flush=True)
