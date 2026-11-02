from shared import *
from network import Predictor,parameter_count,torch
from run import camera_features,infer,forward
from train import seed_all,DEVICE
def main():
 cfg=read(OUT/'protocol.json');oldcfg=read(BASELINE/'protocol.json');data=dict(np.load(DATA/'dataset.npz'));meta=read(DATA/'samples.json');folds=read(OUT/'folds.json');oldfolds=read(BASELINE/'folds.json')
 assert sha(Path(__file__).with_name('network.py'))==sha(ROOT/'scripts/transformer_fusion_20261005/network.py')
 for k in ['seeds','epochs','updates_per_epoch','batch_size','learning_rate','weight_decay','capacity','validation_every']:assert cfg[k]==oldcfg[k]
 for f,o in zip(folds,oldfolds):
  assert f['test_indices']==o['test_indices'] and f['val_indices']==o['val_indices']
  assert [i for i in f['train_indices'] if i<199]==o['train_indices']
  sets=[{meta[i]['recording'] for i in f[k]} for k in ['train_indices','val_indices','test_indices']]
  assert not sets[0]&sets[1] and not sets[0]&sets[2] and not sets[1]&sets[2]
 other=read(VERSION/('old' if ARM=='new' else 'new')/'folds.json')
 assert all(len(a['train_indices'])==len(b['train_indices']) and a['static_added']==b['static_added'] for a,b in zip(folds,other))
 features,rotations=camera_features(data,0,17);idx=np.arange(3);changed={k:v.copy() for k,v in data.items()}
 for k in ['xy','boxes','H','R','cam','seconds']:changed[k][:,P:]=np.random.default_rng(10).normal(size=changed[k][:,P:].shape)
 cf,cr=camera_features(changed,0,17);assert np.array_equal(features,cf) and np.array_equal(rotations,cr);checks=[]
 for variant in VARIANTS:
  seed_all(17);model=Predictor(variant,cfg['capacity'][variant]['ff_width']).to(DEVICE);model.eval();h=history(data,idx,model.geometry);z=forward(model,h,features,idx);z.square().mean().backward()
  assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
  a=infer(model,data,idx,features,rotations);b=infer(model,changed,idx,cf,cr);assert np.array_equal(a['pred'],b['pred'])
  assert parameter_count(model)==cfg['capacity'][variant]['trainable_parameters'];checks.append(dict(variant=variant,connected=True,future_mutation=0))
 write(OUT/'preflight.json',dict(passed=True,checks=checks,equal_training_counts=True,equal_static_counts=True,exact_validation_test=True))
 print(ARM,'preflight passed',flush=True)
if __name__=='__main__':main()
