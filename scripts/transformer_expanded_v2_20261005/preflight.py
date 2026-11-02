"""Before fitting people: disjoint folds, architecture equality, no future input."""
from shared import *
from network import Predictor,parameter_count,torch
from run import camera_features,infer,forward
from train import seed_all,DEVICE

def main():
 assert read(CAMERA/'training_complete.json')['passed'];cfg=read(OUT/'protocol.json');data=dict(np.load(DATA/'dataset.npz'));meta=read(DATA/'samples.json');folds=read(OUT/'folds.json');cfolds=read(CAMERA/'folds.json')
 assert sha(Path(__file__).with_name('network.py'))==sha(ROOT/'scripts/transformer_fusion_20261005/network.py')
 oldfolds=read(BASELINE/'folds.json');oldcfg=read(BASELINE/'protocol.json')
 for key in ['seeds','epochs','updates_per_epoch','batch_size','learning_rate','weight_decay','capacity','validation_every']:
  assert cfg[key]==oldcfg[key]
 for f in folds[:11]:
  assert f['val_indices']==oldfolds[f['fold']]['val_indices']
  assert f['test']==oldfolds[f['fold']]['test'] and f['validation']==oldfolds[f['fold']]['validation']
  assert all(meta[i]['recording']==f['validation'] for i in f['unused_validation_indices'])
 for f,c in zip(folds,cfolds):
  assert f['test']==c['test'] and f['validation']==c['validation'];assert not set(c['train'])&{f['test'],f['validation']}
  sets=[set(meta[i]['recording'] for i in f[k]) for k in ['train_indices','val_indices','test_indices']];assert not sets[0]&sets[1] and not sets[0]&sets[2] and not sets[1]&sets[2]
 features,rotations=camera_features(data,0,17);idx=np.arange(3);changed={k:v.copy() for k,v in data.items()}
 for k in ['xy','boxes','H','R','cam','seconds']:changed[k][:,P:]=np.random.default_rng(10).normal(size=changed[k][:,P:].shape)
 cf,cr=camera_features(changed,0,17);assert np.array_equal(features,cf) and np.array_equal(rotations,cr);checks=[]
 for variant in VARIANTS:
  seed_all(17);model=Predictor(variant,cfg['capacity'][variant]['ff_width']).to(DEVICE);model.eval();h=history(data,idx,model.geometry);z=forward(model,h,features,idx);z.square().mean().backward()
  assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
  a=infer(model,data,idx,features,rotations);b=infer(model,changed,idx,cf,cr);assert np.array_equal(a['pred'],b['pred'])
  assert parameter_count(model)==cfg['capacity'][variant]['trainable_parameters']
  checks.append(dict(variant=variant,connected=True,future_mutation=0.))
 write(OUT/'preflight.json',dict(passed=True,checks=checks,split_recording_disjoint=True,exact_previous_network=True,torch=torch.__version__,device=DEVICE))
 print('Preflight passed on',DEVICE,torch.__version__,flush=True)
if __name__=='__main__':main()
