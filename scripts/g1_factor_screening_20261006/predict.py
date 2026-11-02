"""Frozen held-out inference, full original replay and history-only audit."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
from setup import *
import sys,time
sys.path.insert(0,str(ROOT/'scripts/transformer_matched97_20261006'))
from run import infer,CameraPredictor,tensor,DEVICE
from network import Predictor,torch
from shared import history
torch.set_num_threads(2)

@torch.no_grad()
def camera_inputs(model,data):
 x=tensor(data['cam'][:,:6]);seq,last=model.encoder(x)
 return torch.cat([x,seq],-1).cpu().numpy(),model.head(last[-1]).reshape(-1,15,3).cpu().numpy()

def main():
 begin=time.perf_counter();data=dict(np.load(OUT/'dataset.npz'));meta=read(OUT/'samples.json');original=dict(np.load(BANK/'data/dataset.npz'))
 cfg=read(BANK/'protocol.json');checks=[];camera_checks=[];results={v:[] for v in VARIANTS};components={v:[] for v in VARIANTS};depths={v:[] for v in VARIANTS}
 # Physically remove future suffixes; normal inference must remain identical.
 past={k:(a[:,:6].copy() if k in ['xy','boxes','H','R','cam','seconds'] else a.copy()) for k,a in data.items()}
 for seed in SEEDS:
  seed_result={v:np.zeros((214,15,2)) for v in VARIANTS};seed_component={v:np.zeros((214,15,2)) for v in VARIANTS};seed_depth={v:np.zeros((214,15)) for v in VARIANTS}
  for f in read(OUT/'folds.json'):
   fold=f['fold'];idx=np.array(f['indices']);oldidx=np.array(f['original_test_indices'])
   ck=torch.load(BANK/f'camera/fold_{fold}/seed_{seed}/full/camera.pt',map_location=DEVICE,weights_only=False)
   camera=CameraPredictor().to(DEVICE);camera.load_state_dict(ck['model']);camera.eval()
   of,orr=camera_inputs(camera,original)
   expected=np.load(BANK/f'camera/fold_{fold}/seed_{seed}/full/predictions.npz')['person_rotation']
   er=float(abs(orr-expected).max());assert er<1e-5,er
   feats,rots=camera_inputs(camera,data);pf,pr=camera_inputs(camera,past)
   assert np.array_equal(feats,pf) and np.array_equal(rots,pr)
   camera_checks.append(dict(fold=fold,seed=seed,original_rotation_replay_max=er,future_removed_effect=0.))
   for variant in VARIANTS:
    path=BANK/f'fold_{fold}/seed_{seed}/{variant}'
    ck=torch.load(path/'checkpoint.pt',map_location=DEVICE,weights_only=False)
    assert ck['fold']['test']==f['test'] and f['test'] not in ck['fold']['train']+[ck['fold']['validation']]
    assert ck['protocol_sha256']==sha(BANK/'protocol.json')
    model=Predictor(variant,ck['ff_width']).to(DEVICE);model.load_state_dict(ck['model']);model.eval()
    old=infer(model,original,oldidx,of,orr)
    cached=np.load(path/'predictions_reviewed.npz');assert np.array_equal(cached['indices'],oldidx)
    replay=max(float(abs(old[k]-cached[k]).max()) for k in ['pred','component','input_xy'])
    assert replay<1e-4,(variant,fold,seed,replay)
    z=infer(model,data,idx,feats,rots);zp=infer(model,past,idx,pf,pr)
    future=max(float(abs(z[k]-zp[k]).max()) for k in ['pred','component','input_xy'])
    assert future==0,(variant,fold,seed,future)
    assert np.isfinite(z['pred']).all() and np.isfinite(z['component']).all()
    seed_result[variant][idx]=z['pred'];seed_component[variant][idx]=z['component'];seed_depth[variant][idx]=z['depth']
    checks.append(dict(fold=fold,seed=seed,variant=variant,windows=len(idx),original_replay_max_px=replay,future_removed_effect_px=future))
   print('seed',seed,'fold',fold,'windows',len(idx),'six models verified',flush=True)
  for v in VARIANTS:results[v].append(seed_result[v]);components[v].append(seed_component[v]);depths[v].append(seed_depth[v])
 for v in VARIANTS:np.savez_compressed(OUT/f'predictions_{v}.npz',pred=np.stack(results[v]),component=np.stack(components[v]),depth=np.stack(depths[v]),seeds=SEEDS)
 hold=np.repeat(data['xy'][:,5,None],15,axis=1)
 np.savez_compressed(OUT/'predictions_hold_last.npz',pred=hold)
 protected=read(OUT/'protected_sources.json');changed=[p for p,h in protected.items() if sha(ROOT/p)!=h];assert not changed,changed
 write(OUT/'inference_audit.json',dict(passed=True,models=len(checks),camera_models=len(camera_checks),predicted_window_seed_model_count=sum(c['windows'] for c in checks),seconds=time.perf_counter()-begin,checks=checks,camera_checks=camera_checks,protected_files=len(protected),changed_files=changed,training=False))
 print('Inference complete:',len(checks),'models;',time.perf_counter()-begin,'seconds',flush=True)
if __name__=='__main__':main()
