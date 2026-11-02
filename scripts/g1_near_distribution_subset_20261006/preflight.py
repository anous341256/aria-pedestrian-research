from shared import *
from network import Predictor,parameter_count,torch
from run import camera_features,infer,forward
from train import seed_all,DEVICE

def main():
    cfg=read(OUT/'protocol.json');oldcfg=read(BASELINE/'protocol.json');data=dict(np.load(DATA/'dataset.npz'));meta=read(DATA/'samples.json')
    folds=read(OUT/'folds.json');oldfolds=read(BASELINE/'folds.json');selection=read(VERSION/'selection_locked.json');design=read(VERSION/'design_locked.json')
    assert sha(Path(__file__).with_name('network.py'))==sha(ROOT/'scripts/transformer_matched97_20261006/network.py')
    for rel,digest in read(VERSION/'selection_hashes.json').items():assert sha(VERSION/rel)==digest
    for k in ['seeds','epochs','updates_per_epoch','batch_size','learning_rate','weight_decay','capacity','validation_every']:assert cfg[k]==oldcfg[k]
    for f,o,entry in zip(folds,oldfolds,selection['folds']):
        assert f['test_indices']==o['test_indices'] and f['val_indices']==o['val_indices']
        assert set(f['train_indices'])<=set(o['train_indices']) & set(design['eligible'])
        assert len(f['train_indices'])==entry['selected'] and len(set(f['train_indices']))==entry['selected']
        sets=[{meta[i]['recording'] for i in f[k]} for k in ['train_indices','val_indices','test_indices']]
        assert not sets[0]&sets[1] and not sets[0]&sets[2] and not sets[1]&sets[2]
        assert sorted(sets[0])==sorted({meta[i]['recording'] for i in f['eligible_train_indices']})
    features,rotations=camera_features(data,0,17);idx=np.arange(3);changed={k:v.copy() for k,v in data.items()}
    for k in ['xy','boxes','H','R','cam','seconds']:changed[k]=changed[k][:,:P].copy()
    cf,cr=camera_features(changed,0,17);assert np.array_equal(features,cf) and np.array_equal(rotations,cr)
    checks=[]
    for v in VARIANTS:
        seed_all(17);model=Predictor(v,cfg['capacity'][v]['ff_width']).to(DEVICE);model.eval();h=history(data,idx,model.geometry)
        forward(model,h,features,idx).square().mean().backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        a=infer(model,data,idx,features,rotations);b=infer(model,changed,idx,cf,cr);assert np.array_equal(a['pred'],b['pred'])
        assert parameter_count(model)==cfg['capacity'][v]['trainable_parameters'];checks.append(dict(variant=v,connected=True,future_truncation_difference=0))
    write(OUT/'preflight.json',dict(passed=True,checks=checks,exact_validation_test=True,subset_provenance=True,selection_hashes=True))
    print(ARM,'preflight passed',flush=True)
if __name__=='__main__':main()
