"""Paired old199 evaluation and independent geometry / preservation audit."""
from shared import *
import collections,importlib.util
def main():
 summaries={a:read(VERSION/a/'summary.json') for a in ['old','new']};records={a:[r for r in read(VERSION/a/'metrics.json') if r['mode']=='reviewed' and r['method'] in VARIANTS] for a in summaries}
 spec=importlib.util.spec_from_file_location('motion_screen',ROOT/'scripts/sam2_g1_matched97_20261006/window_motion.py');screen=importlib.util.module_from_spec(spec);spec.loader.exec_module(screen)
 base=dict(np.load(BASELINE/'data/dataset.npz'));same_screen={i for i in range(199) if screen.eligible(screen.rotation_stats(base['R'][i],base['seconds'][i]))}
 results={}
 for v in VARIANTS:
  maps={a:{(r['key'],r['seed']):r for r in records[a] if r['method']==v} for a in summaries};assert maps['old'].keys()==maps['new'].keys()
  diffs=[dict(key=k[0],seed=k[1],recording=maps['old'][k]['recording'],delta_ADE=maps['new'][k]['metrics']['ADE15']-maps['old'][k]['metrics']['ADE15'],delta_FDE=maps['new'][k]['metrics']['FDE15']-maps['old'][k]['metrics']['FDE15']) for k in maps['old']]
  byrec={r:float(np.mean([d['delta_ADE'] for d in diffs if d['recording']==r])) for r in sorted({d['recording'] for d in diffs})}
  bywin={k:float(np.mean([d['delta_ADE'] for d in diffs if d['key']==k])) for k in sorted({d['key'] for d in diffs})}
  results[v]=dict(old=summaries['old']['reviewed'][v],new=summaries['new']['reviewed'][v],delta_ADE=float(np.mean([d['delta_ADE'] for d in diffs])),delta_FDE=float(np.mean([d['delta_FDE'] for d in diffs])),per_seed_delta={str(s):float(np.mean([d['delta_ADE'] for d in diffs if d['seed']==s])) for s in [17,29,43]},per_recording_delta=byrec,recording_wins=sum(x<0 for x in byrec.values()),window_wins=sum(x<0 for x in bywin.values()),sparse={a:summaries[a]['sparse'][v] for a in summaries},base199=read(BASELINE/'summary.json')['reviewed'][v])
  results[v]['same_motion_screen_test']={a:{k:float(np.mean([r['metrics'][k] for r in records[a] if r['method']==v and r['index'] in same_screen])) for k in ['ADE15','FDE15']} for a in summaries}
 distribution={}
 for arm in summaries:
  ds=dict(np.load(VERSION/arm/'data/dataset.npz'));mm=read(VERSION/arm/'data/samples.json');idx=np.arange(199,296);turn=np.degrees(np.linalg.norm(ds['cam'][idx,-1],axis=1));h=(ds['boxes'][idx,P-1,3]-ds['boxes'][idx,P-1,1])/1920
  distribution[arm]=dict(windows=97,static=sum(m.get('static_control',False) for m in mm[199:]),recordings=dict(collections.Counter(m['recording'] for m in mm[199:])),clips=len({m['clip'] for m in mm[199:]}),tracks=len({(m['clip'],m['id']) for m in mm[199:]}),turn_mean_deg=float(turn.mean()),turn_ge15_fraction=float(np.mean(turn>=15)),height_mean=float(h.mean()))
 # Independent inverse-rotation pinhole composition, not the runtime projector.
 from scipy.spatial.transform import Rotation
 geometry=[]
 for arm in summaries:
  data=dict(np.load(VERSION/arm/'data/dataset.npz'))
  for fold in read(VERSION/arm/'folds.json'):
   f=fold['fold'];idx=np.array(fold['test_indices'])
   for seed in [17,29,43]:
    op=VERSION/'old'/f'camera/fold_{f}/seed_{seed}/full/camera.pt';npth=VERSION/'new'/f'camera/fold_{f}/seed_{seed}/full/camera.pt';assert sha(op)==sha(npth)
    rv=np.load(VERSION/arm/f'camera/fold_{f}/seed_{seed}/full/predictions.npz')['person_rotation'][idx]
    for variant in [VARIANTS[2],VARIANTS[5]]:
     z=np.load(VERSION/arm/f'fold_{f}/seed_{seed}/{variant}/predictions_reviewed.npz');errors=[]
     for j,i in enumerate(idx):
      K=data['K'][i].astype(float);rays=np.concatenate([z['component'][j],np.ones((F,1))],-1)@np.linalg.inv(K).T
      future=np.einsum('tji,tj->ti',Rotation.from_rotvec(rv[j]).as_matrix(),rays);uv=future@K.T;xy=uv[:,:2]/uv[:,2,None]
      errors.append(float(np.max(np.abs(xy-z['pred'][j]))))
     assert max(errors)<.01,(arm,f,variant,max(errors));geometry.append(dict(arm=arm,fold=f,seed=seed,variant=variant,max_px=max(errors)))
 protected=read(VERSION/'protected_previous_hashes.json');assert all(sha(ROOT/p)==h for p,h in protected.items())
 write(VERSION/'comparison.json',dict(results=results,distribution=distribution,hold_last=summaries['new']['reviewed']['hold_last'],protocol=read(VERSION/'selection_locked.json'),same_motion_screen_test_indices=sorted(same_screen)))
 write(VERSION/'paired_audit.json',dict(passed=True,models_verified=sum(read(VERSION/a/'verification.json')['models'] for a in summaries),same_cameras=33,geometry=geometry,geometry_max_px=max(x['max_px'] for x in geometry),protected_files=len(protected)))
 for v,r in results.items():print(v,'old',round(r['old']['window_mean']['ADE15'],2),'new',round(r['new']['window_mean']['ADE15'],2),'delta',round(r['delta_ADE'],2),'seeds',r['per_seed_delta'],flush=True)
if __name__=='__main__':main()
