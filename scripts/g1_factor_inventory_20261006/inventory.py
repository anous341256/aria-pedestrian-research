"""Read-only inventory of reviewed cohorts; never loads forecast errors."""
from pathlib import Path
import json,gzip,hashlib,collections
import numpy as np,pandas as pd
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/g1_factor_inventory_20261006'
OUT.mkdir(parents=True,exist_ok=True)
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf8')
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
INV={r['id']:r for r in read(ROOT/'outputs/dataset_threeway_20261001/motion_inventory.json')}
FACTORS=list('ABCDEFGH')
POSE={};CLIPS={}
def clip_paths(cid):
 n=int(cid)
 v='sam2_near_people_20261002' if n<=10 else 'sam2_g1_expansion_20261005' if n<=21 else 'sam2_g1_expansion_v2_20261005' if n<=31 else 'sam2_g1_matched97_20261006'
 folder=ROOT/'outputs'/v/cid
 inp=ROOT/'outputs/sam2_g1_v6_v8_20261001'/cid/'input' if n<=10 else folder/'input'
 return folder,inp
def clip(cid):
 if cid not in CLIPS:
  folder,inp=clip_paths(cid);cal=dict(np.load(inp/'reference.npz'))
  with gzip.open(folder/'final/masks_rle.jsonl.gz','rt',encoding='utf8') as f: masks={(r['id'],r['frame']):r for r in map(json.loads,f)}
  CLIPS[cid]=(cal,masks,read(inp/'alignment.json'),folder,inp)
 return CLIPS[cid]
def pose(sid):
 if sid not in POSE:
  p=Path(INV[sid]['mps_source']);df=pd.read_csv(p)
  POSE[sid]=(np.rint(df.tracking_timestamp_us.to_numpy()*1000).astype(np.int64),df)
 return POSE[sid]
def angle(a,b):
 q=np.swapaxes(a,-1,-2)@b
 return np.degrees(np.arccos(np.clip((np.trace(q,axis1=-2,axis2=-1)-1)/2,-1,1)))
def calc(m,arr):
 cal,masks,align,folder,inp=clip(m['clip']);fs=np.asarray(m['frames']);ns=cal['times_ns'][fs];t=(ns-ns[5])/1e9; R=cal['R'][fs];b=arr['boxes'].astype(float)
 reasons=[]
 if not np.all((np.diff(t)>.08)&(np.diff(t)<.12)):reasons.append('time_step')
 if not all(align[f]['status']=='valid' and align[f]['quality']>=.5 for f in fs) or len({align[f]['graph'] for f in fs})!=1:reasons.append('pose_alignment')
 for f in fs:
  r=masks.get((m['id'],int(f)))
  if r is None:reasons.append('missing_final_mask');continue
  q=r['quality']
  if q['guide_iou']<.45 or q['outside']>.1:reasons.append('mask_guide_quality')
 if np.any(b[:,0]<=5)|np.any(b[:,1]<=5)|np.any(b[:,2]>=2555)|np.any(b[:,3]>=1915):reasons.append('border')
 if np.any(b[:,3]-b[:,1]<.08*1920):reasons.append('retention_scale')
 mx=float(max(angle(R[:-10],R[10:])));om=float(np.percentile(angle(R[:-1],R[1:])/np.diff(t),95))
 if mx>20 or om>50:reasons.append('g1_camera_screen')
 sid=m['recording'].split('_')[-1];ts,df=pose(sid);idx=np.searchsorted(ts,ns).clip(1,len(ts)-1);idx=np.where(abs(ts[idx]-ns)<abs(ts[idx-1]-ns),idx,idx-1)
 rows=df.iloc[idx];gap=np.abs(ts[idx]-ns)/1e6
 if gap.max()>30 or rows.quality_score.min()<.5 or rows.graph_uid.nunique()!=1:reasons.append('translation_pose_alignment')
 rd=Rotation.from_quat(rows[['qx_world_device','qy_world_device','qz_world_device','qw_world_device']].to_numpy()).as_matrix()
 device=rows[['tx_world_device','ty_world_device','tz_world_device']].to_numpy()
 pc=device+np.einsum('nij,j->ni',rd,cal['T_device_camera'][:3,3])
 if not np.isfinite(pc).all() or max(np.linalg.norm(np.diff(pc,axis=0),axis=1)/np.diff(t))>5:reasons.append('translation_jump')
 xy=arr['xy'].astype(float);H=arr['H'].astype(float);p=np.einsum('nij,nj->ni',H,np.c_[xy,np.ones(21)]);p=p[:,:2]/p[:,2:]
 def feature(k):
  tt=t[:k]-t[:k].mean();height=b[:k,3]-b[:k,1];den=np.sum(tt**2)
  vel=np.sum(tt[:,None]*p[:k],axis=0)/den
  fit=p[:k].mean(0)+tt[:,None]*vel
  return [float(angle(R[:k-1],R[1:k]).sum()),float(angle(R[0],R[k-1])),float(np.linalg.norm(np.diff(pc[:k],axis=0),axis=1).sum()/(t[k-1]-t[0])),float(np.median(height)/1920),float(np.sum(tt*np.log(height))/den),float(vel[0]/2560),float(vel[1]/1920),float(np.sqrt(np.mean(np.sum((p[:k]-fit)**2,axis=1)))/np.median(height))]
 f=feature(6);f_full=feature(21)
 # Delete-one interior historical point sensitivity, never using future outcomes.
 stability={};hist=np.arange(6);base_t=t[:6]
 variants=[]
 for drop in [1,2,3,4]:
  ids=hist[hist!=drop];tt=base_t[ids]-base_t[ids].mean();den=np.sum(tt**2);h=b[ids,3]-b[ids,1];v=np.sum(tt[:,None]*p[ids],axis=0)/den;fit=p[ids].mean(0)+tt[:,None]*v
  variants.append([float(np.median(h)/1920),float(np.sum(tt*np.log(h))/den),float(v[0]/2560),float(v[1]/1920),float(np.sqrt(np.mean(np.sum((p[ids]-fit)**2,axis=1)))/np.median(h))])
 for j,key in enumerate('DEFGH'):stability[key]=float(np.max(np.abs(np.array(variants)[:,j]-f[3+j])))
 return dict(key=m['key'],clip=m['clip'],id=m['id'],recording=m['recording'],source_version=m.get('source_version','unknown'),start_s=float(m['source_start_s']+fs[0]/10),end_s=float(m['source_start_s']+fs[-1]/10),cutoff=m['cutoff'],frames=fs.tolist(),features=dict(zip(FACTORS,f)),full_window_features=dict(zip(FACTORS,f_full)),sensitivity=stability,eligible=not reasons,reasons=sorted(set(reasons)),camera_screen=dict(max_1s_rotation=mx,omega_p95=om),input=str(inp/'rgb.mp4'),annotation=str(folder/'final'),static_control=bool(m.get('static_control',False)),visual_g1_status='Previously sampled reviewed; current numeric screen only, per-window visual verification pending')
def pack_nonoverlap(rows,by_person=True):
 result=[];last={}
 for r in sorted(rows,key=lambda r:(r['recording'],r['end_s'],r['start_s'],r['key'])):
  group=(r['recording'],r['clip'],r['id']) if by_person else r['recording']
  if r['start_s']>last.get(group,-1)+.049:result.append(r);last[group]=r['end_s']
 return result
def summarize(rows,threshold):
 matrix=read(ROOT/'plans/G1_FACTOR_SCREENING_DESIGN_20261006.json')['rows'];counts=[]
 for i,sign in enumerate(matrix):
  pool=[r for r in rows if r.get('levels')==sign];person=pack_nonoverlap(pool);time=pack_nonoverlap(pool,False)
  counts.append(dict(cell=i+1,levels=sign,windows=len(pool),person_nonoverlap=len(person),recording_time_nonoverlap=len(time),recordings=len({r['recording'] for r in pool}),tracks=len({(r['clip'],r['id']) for r in pool}),keys=[r['key'] for r in person]))
 arr=np.array([[r['features'][k] for k in FACTORS] for r in rows]);lv=np.array([r['levels'] for r in rows])
 return dict(windows=len(rows),person_nonoverlap=len(pack_nonoverlap(rows)),recording_time_nonoverlap=len(pack_nonoverlap(rows,False)),recordings=len({r['recording'] for r in rows}),tracks=len({(r['clip'],r['id']) for r in rows}),cells=counts,feature_correlation=np.corrcoef(arr.T).tolist(),design_main_rank=int(np.linalg.matrix_rank(np.c_[np.ones(len(lv)),lv])),thresholds=threshold,oa_assigned=sum(c['windows'] for c in counts),level_disagreement_under_delete_one={k:sum((r['features'][k]-threshold[k])**2<=r['sensitivity'][k]**2 for r in rows) for k in 'DEFGH'})
def main():
 pools=[(ROOT/'outputs/transformer_expanded_v2_20261005/data/dataset.npz',ROOT/'outputs/transformer_expanded_v2_20261005/data/samples.json'),(ROOT/'outputs/sam2_g1_matched97_20261006/dataset/all_eligible_dataset.npz',ROOT/'outputs/sam2_g1_matched97_20261006/dataset/all_eligible_samples.json')]
 out=[];seen=set();protection={}
 for dp,mp in pools:
  data=dict(np.load(dp));meta=read(mp)
  for p in [dp,mp]:protection[str(p.relative_to(ROOT))]=digest(p)
  for i,m in enumerate(meta):
   assert m['key'] not in seen;seen.add(m['key']);out.append(calc(m,{k:v[i] for k,v in data.items()}))
 good=[r for r in out if r['eligible']];ind=pack_nonoverlap(good)
 thresholds={k:float(np.median([r['features'][k] for r in ind])) for k in FACTORS}
 lock=OUT/'thresholds_locked.json'
 if lock.exists():assert read(lock)['thresholds']==thresholds
 else:write(lock,dict(thresholds=thresholds,source_keys=[r['key'] for r in ind],rule='Median on existing, numerically screened, per-person nonoverlapping development windows, before expansion or forecast error access. Frozen for new data.',independent_recording_warning='Nonoverlap does not establish independent people or recordings.'))
 for r in out:r['levels']=[1 if r['features'][k]>thresholds[k] else -1 for k in FACTORS]
 write(OUT/'existing_windows.json',out);summary=summarize(good,thresholds);summary['excluded']=dict(collections.Counter(z for r in out for z in r['reasons']));summary['total_source_windows']=len(out)
 for cid,(_,_,_,folder,inp) in CLIPS.items():
  for p in [folder/'final/masks_rle.jsonl.gz',folder/'final/census.json',inp/'reference.npz']:protection[str(p.relative_to(ROOT))]=digest(p)
 write(OUT/'existing_summary.json',summary);write(OUT/'protected_sources.json',protection)
 print(json.dumps({k:v for k,v in summary.items() if k not in ['feature_correlation','cells']},indent=2))
 print('CELLS',[(r['cell'],r['windows'],r['person_nonoverlap'],r['recordings']) for r in summary['cells']])
if __name__=='__main__':main()
