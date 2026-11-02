"""Independent SciPy projection, checkpoint replay and paired model attribution."""
import os
from pathlib import Path
import json,hashlib,sys
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from network import ConditionalDiT
from experiment import ROOT,OUT,OLD,BASE,CAM,read,write,sha,fixed_noise,camera_keys,DEVICE,input_history,targets,SCALE,P,F
from sampling import infer

def errors(pred,truth,depth=None):
 e=np.linalg.norm(pred-truth,axis=-1)
 bad=~np.isfinite(e)
 if depth is not None:bad|=depth<=1e-6
 e[bad]=32000.
 return e,bad

def independent_project(points,rv,K):
 rot=Rotation.from_rotvec(rv.reshape(-1,3)).as_matrix().reshape(*rv.shape[:-1],3,3)
 # camera-to-reference relative rotation, transposed for reference-to-camera projection.
 H=K[:,None]@np.swapaxes(rot,-1,-2)@np.linalg.inv(K)[:,None]
 hom=np.concatenate([points,np.ones((*points.shape[:-1],1))],-1)
 q=np.einsum('btij,btj->bti',H,hom)
 return q[...,:2]/q[...,2:],q[...,2]

def load(branch,fold):
 m=ConditionalDiT(3,3) if branch=='camera' else ConditionalDiT(6,2,2)
 saved=torch.load(OUT/f'fold_{fold}/{branch}/checkpoint.pt',map_location=DEVICE,weights_only=False)
 assert saved['protocol_sha256']==sha(OUT/'protocol.json')
 m=m.to(DEVICE);m.load_state_dict(saved['model']);m.eval();return m

def angular(a,b):
 ra=Rotation.from_rotvec(a.reshape(-1,3));rb=Rotation.from_rotvec(b.reshape(-1,3))
 return np.rad2deg((ra.inv()*rb).magnitude()).reshape(a.shape[:-1])

def main():
 cfg=read(OUT/'protocol.json');data=dict(np.load(OLD/'dataset.npz'));meta=read(OLD/'samples.json')
 cf=read(CAM/'folds.json');full=np.load(CAM/'dataset.npz')['rotation']
 report={};audit=[];rows=[];covered=[];maxgeom=0.;camera_error={k:[] for k in ['old','ddim0','ddim1']}
 for fold in read(OUT/'folds.json'):
  f=fold['fold'];r=dict(np.load(OUT/f'fold_{f}/predictions.npz'));idx=r['indices'];covered.extend(idx.tolist())
  h=input_history(data,idx)
  # Physically remove all future time suffixes before producing inference inputs.
  short=dict(data)
  for key in ['xy','boxes','H','R','cam','seconds']:short[key]=data[key][:,:P].copy()
  hs=input_history(short,idx)
  assert all(np.array_equal(h[k],hs[k]) for k in h)
  m=load('person',f);c=load('camera',f)
  # Independently check fitted normalization against the allowed training set.
  tr=fold['train_indices'];th=input_history(data,tr);ty=targets(data,tr,th['origin'])
  expected=ty['future'].reshape(-1,F,2).mean((0,1))
  assert np.allclose(m.target_mean[0].detach().cpu().numpy(),expected,atol=1e-6)
  ctr=cf[f]['train_indices'];assert np.allclose(c.target_mean[0].detach().cpu().numpy(),full[ctr,P:].mean((0,1)),atol=1e-5)
  noise=fixed_noise(['person:'+meta[i]['key'] for i in idx],2)
  for name,eta in [('ddim0',0.),('ddim1',1.)]:
   ps=infer(m,hs['seq'],hs['origin'],noise,eta=eta,steps=cfg['sampling_steps'])
   cs=infer(c,short['cam'][idx],eta=eta,steps=cfg['sampling_steps'])
   assert np.array_equal(ps,r[name+'_person_samples'])
   assert np.array_equal(cs,r[name+'_camera_samples'])
   pm=(ps.mean(1)+h['origin'][:,None])*SCALE;rv=cs.mean(1)
   point,depth=independent_project(pm,rv,data['K'][idx]);delta=float(np.max(np.abs(point-r[name])))
   maxgeom=max(maxgeom,delta);assert delta<.02,(f,name,delta)
   camera_error[name].extend(angular(rv,data['cam'][idx,P:]).tolist())
   # Diversity and oracle-best metrics remain secondary, same K and invalid penalty.
   all_errors=[]
   for k in range(cfg['samples']):
    jp,jd=independent_project((ps[:,k]+h['origin'][:,None])*SCALE,cs[:,k],data['K'][idx])
    je,_=errors(jp,r['truth'],jd);all_errors.append(je)
   all_errors=np.stack(all_errors,1)
   for j,i in enumerate(idx):
    rows.append(dict(index=int(i),key=meta[i]['key'],recording=meta[i]['recording'],method=name+'_samples',
     expected_ADE=float(all_errors[j].mean()),minADE5=float(all_errors[j].mean(-1).min()),minFDE5=float(all_errors[j,:,-1].min())))
  # Original camera/person recomposition reproduces the saved old result.
  op,od=independent_project(r['old_component'],r['old_camera'],data['K'][idx]);assert np.max(np.abs(op-r['old_prediction']))<.02
  camera_error['old'].extend(angular(r['old_camera'],data['cam'][idx,P:]).tolist())
  arms=['old_prediction','ddim0','ddim1','new_person_old_camera','old_person_new_camera','new_person_GT_camera']
  for name in arms+['hold_last']:
   point=np.repeat(data['xy'][idx,P-1:P],F,axis=1) if name=='hold_last' else r[name]
   depth=od if name=='old_prediction' else r.get(name+'_depth')
   e,bad=errors(point,r['truth'],depth)
   for j,i in enumerate(idx):rows.append(dict(index=int(i),key=meta[i]['key'],recording=meta[i]['recording'],method=name,ADE=float(e[j].mean()),FDE=float(e[j,-1]),invalid=int(bad[j].sum())))
  truth_component=targets(data,idx,h['origin'])['stable'][:,P:]
  for name,point in [('old_component',r['old_component']),('new_component',r['ddim0_component'])]:
   e,bad=errors(point,truth_component)
   for j,i in enumerate(idx):rows.append(dict(index=int(i),key=meta[i]['key'],recording=meta[i]['recording'],method=name,ADE=float(e[j].mean()),FDE=float(e[j,-1]),invalid=int(bad[j].sum())))
  audit.append(dict(fold=f,checkpoint_replay_exact=True,physical_future_truncation_exact=True,training_normalization=True))
 assert sorted(covered)==list(range(117))
 for method in sorted({r['method'] for r in rows}):
  rr=[r for r in rows if r['method']==method];recordings=sorted({r['recording'] for r in rr})
  if 'ADE' in rr[0]:
   report[method]=dict(ADE=float(np.mean([r['ADE'] for r in rr])),FDE=float(np.mean([r['FDE'] for r in rr])),recording_equal_ADE=float(np.mean([np.mean([r['ADE'] for r in rr if r['recording']==g]) for g in recordings])),invalid=sum(r['invalid'] for r in rr))
  else:report[method]={k:float(np.mean([r[k] for r in rr])) for k in ['expected_ADE','minADE5','minFDE5']}
 for method,values in camera_error.items():
  a=np.array(values);report[method+'_camera']=dict(mean_degrees=float(a.mean()),final_degrees=float(a[:,-1].mean()))
 protected=read(OUT/'source_hashes.json');assert all(sha(ROOT/p)==s for p,s in protected.items())
 write(OUT/'metrics.json',rows);write(OUT/'summary.json',report)
 write(OUT/'verification.json',dict(folds=audit,protected_files_unchanged=len(protected),max_independent_projection_error_px=maxgeom,held_out_windows=117,model_seed=17))
 lines=['Dual DiT diffusion V1 — 2026-10-07','',
 'Both camera and pedestrian branches now use conditional DiT v-prediction diffusion. Historical encoder is a Transformer, no GRU in either new branch.',
 '117 held-out windows, six recording folds, seed 17 only. Camera: full allowed MPS windows, 2000 updates. Pedestrian: 240 updates with history reconstruction weight 0.5.',
 'Primary: 20-step deterministic DDIM eta=0, mean of five fixed draws per branch followed by geometric projection. Secondary: stochastic DDIM eta=1, same weights and step count.',
 'This implements discrete samplers, not continuous reverse-SDE / probability-flow-ODE solvers. Network and diffusion choices changed together; not a pure diffusion-only matched-capacity ablation.',
 '', 'Method | ADE px | FDE px | recording-equal ADE px']
 for name in ['old_prediction','hold_last','ddim0','ddim1','new_person_old_camera','old_person_new_camera','new_person_GT_camera','old_component','new_component']:
  r=report[name];lines.append(f'{name} | {r["ADE"]:.2f} | {r["FDE"]:.2f} | {r["recording_equal_ADE"]:.2f}')
 lines+=['','Camera error on the same person windows (degrees):']
 for name in ['old_camera','ddim0_camera','ddim1_camera']:lines.append(name+': '+json.dumps(report[name]))
 lines+=['','All six checkpoint replays and physically future-truncated inference matched exactly. Protected sources unchanged.',
 'Full per-window metrics, all 5 trajectory samples, and camera/person swap diagnostics are saved. minADE5/minFDE5 are oracle diagnostics, not headline point prediction.',
 'No weights or sampling hyperparameters were selected using test scores. One seed and small, overlapping offline reference windows do not establish robustness. Prior selected baseline remains unchanged.',
 '', 'References: https://arxiv.org/abs/2011.13456 ; https://arxiv.org/abs/2010.02502 ; https://arxiv.org/abs/2212.09748']
 dest=ROOT/'reports/dual_dit_diffusion_20261007';dest.mkdir(parents=True,exist_ok=True);(dest/'RESULTS.txt').write_text('\n'.join(lines),encoding='utf-8')
 print(json.dumps(report,indent=2))

if __name__=='__main__':main()
