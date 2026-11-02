from assoc_common import *
from tracker import Tracker
import time,collections
def load_clip(cid):
 inv=read(OUT/'inventory.json');entries=[e for e in inv['entries'] if e['clip']==cid];p=dict(np.load(OUT/'poses'/f'{cid}.npz'));a=np.load(OUT/'appearance'/f'{cid}.npz');emb=dict(zip(a['uids'],a['features']));frames=[]
 for e in entries:
  depth={r['target_key']:r for r in read(OUT/'depth'/e['key']/'measurements.json')} if e['targets'] else {};obs=[]
  for t in e['targets']:
   uid=t['target_key'];r=depth[uid];b=np.asarray(t['box'],float);obs.append(dict(uid=uid,box=b,uv=(b[:2]+b[2:])/2,embedding=emb[uid].astype(float),z=r['z_median_m'] if r['usable'] else 0.,iqr=r['z_iqr_m'] if r['usable'] else 0.,depth_valid=r['usable']))
  frames.append(obs)
 return p,frames
def infer(p,frames,arm,condition):
 tr=Tracker(arm,p['K'],p['R'][0]);rows=[];logs=[]
 for f,obs in enumerate(frames):
  if condition=='blackout3' and f>=20 and f%20<3:obs=[]
  rr,ll=tr.step(obs,p['R'][f],p['c'][f],float((p['times_ns'][f]-p['times_ns'][0])/1e9),f);rows+=rr;logs.append(ll)
 return rows,logs
def main():
 inv=read(OUT/'inventory.json');hashes={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')};write(OUT/'execution_code_hashes.json',hashes);start=time.monotonic()
 for cl in inv['clips']:
  cid=cl['clip'];p,frames=load_clip(cid)
  for condition in ['natural','blackout3']:
   for arm in ARMS:
    rows,logs=infer(p,frames,arm,condition);dest=OUT/'tracking'/condition/arm;write(dest/f'{cid}.json',dict(rows=rows,frames=logs));print(condition,arm,cid,len(rows),len({r['track_id'] for r in rows}),flush=True)
 write(OUT/'tracking_complete.json',dict(passed=True,clip_arm_runs=60,seconds=time.monotonic()-start,code_hashes=hashes))
if __name__=='__main__':main()
