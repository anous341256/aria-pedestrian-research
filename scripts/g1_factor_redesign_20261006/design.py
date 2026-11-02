"""Design from frozen feature support only; no forecast outcomes are read."""
from pathlib import Path
import json,itertools,hashlib,collections
import numpy as np
from scipy.optimize import linprog,milp,Bounds,LinearConstraint
ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'outputs/g1_factor_inventory_20261006'
OUT=ROOT/'outputs/g1_factor_redesign_20261006'
OUT.mkdir(exist_ok=True,parents=True)
F=list('ABCDEFGH')
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf8')
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def load():
 allowed=set(read(SOURCE/'checkpoint_availability.json')['matched97_new']['eligible_keys'])
 return sorted([r for r in read(SOURCE/'combined_windows.json') if r['eligible'] and r['key'] in allowed],key=lambda r:r['key'])

def moment_matrix(x):return np.array([*x.T,*[x[:,i]*x[:,j] for i,j in itertools.combinations(range(x.shape[1]),2)]])

def conflicts(rows):
 return [(i,j) for i,a in enumerate(rows) for j,b in enumerate(rows) if j>i and a['recording']==b['recording'] and max(a['start_s'],b['start_s'])<=min(a['end_s'],b['end_s'])+.049]

def probe():
 rows=load();x=np.array([r['levels'] for r in rows]);m=moment_matrix(x);n=len(rows)
 lp=linprog(np.zeros(n),A_eq=np.r_[np.ones((1,n)),m],b_eq=np.r_[1,np.zeros(len(m))],bounds=(0,None),method='highs')
 print('Exact pairwise balance with unrestricted nonnegative weights:',lp.message,flush=True)
 result=dict(n=n,unique_patterns=len(set(map(tuple,x))),weighted_exact_status=int(lp.status),weighted_exact_message=lp.message)
 if lp.success:result['nonzero_weights']=int((lp.x>1e-8).sum())
 write(OUT/'support_probe.json',result)
 for f in range(8):
  tab={r:collections.Counter(int(x[i,f]) for i,a in enumerate(rows) if a['recording']==r) for r in sorted({r['recording'] for r in rows})}
  print(F[f],dict(collections.Counter(x[:,f])),{k.split('_')[-1]:dict(v) for k,v in tab.items()},flush=True)
 # Exact moments on a 32-window, recording-time nonoverlap design. This is
 # feasibility, not an attempted statement about arbitrary population support.
 pairs=conflicts(rows);recs=sorted({r['recording'] for r in rows});cons=[np.ones(n)];lo=[32];hi=[32]
 for v in m:cons.append(v);lo.append(0);hi.append(0)
 for i,j in pairs:
  v=np.zeros(n);v[i]=v[j]=1;cons.append(v);lo.append(-np.inf);hi.append(1)
 for rec in recs:
  cons.append(np.array([r['recording']==rec for r in rows],float));lo.append(1);hi.append(8)
 res=milp(np.zeros(n),integrality=np.ones(n),bounds=Bounds(0,1),constraints=LinearConstraint(np.array(cons),lo,hi),options=dict(time_limit=40,mip_rel_gap=0))
 print('Exact32:',res.message,flush=True)
 write(OUT/'exact32_probe.json',dict(status=int(res.status),message=res.message,selected=[] if res.x is None else [rows[i]['key'] for i in np.flatnonzero(res.x>.5)]))

if __name__=='__main__':probe()
