"""Audit matched settings, source identity and historical-only inference before fitting."""
from shared import *
from network import Predictor,parameter_count,torch
from run import camera_features,infer,forward
from train import seed_all,DEVICE

def main():
    cfg=read(OUT/'protocol.json');oldcfg=read(BASELINE/'protocol.json')
    data=dict(np.load(DATA/'dataset.npz'));old=dict(np.load(BASELINE/'data/dataset.npz'))
    meta=read(DATA/'samples.json');folds=read(OUT/'folds.json');oldfolds=read(BASELINE/'folds.json')
    assert sha(Path(__file__).with_name('network.py'))==sha(ROOT/'scripts/transformer_matched97_20261006/network.py')
    for k in ['seeds','epochs','updates_per_epoch','batch_size','learning_rate','weight_decay','capacity','validation_every','evaluation_inputs','loss','selection']:assert cfg[k]==oldcfg[k],k
    assert meta[:296]==read(BASELINE/'data/samples.json')
    for k in old:assert np.array_equal(data[k][:296],old[k]),k
    assert len(folds)==11
    for f,o in zip(folds,oldfolds):
        assert f['test_indices']==o['test_indices'] and f['val_indices']==o['val_indices']
        assert [i for i in f['train_indices'] if i<296]==o['train_indices']
        assert f['new_train_indices']==[i for i in range(296,len(meta)) if meta[i]['recording'] not in [f['test'],f['validation']]]
        groups=[{meta[i]['recording'] for i in f[k]} for k in ['train_indices','val_indices','test_indices']]
        assert not groups[0]&groups[1] and not groups[0]&groups[2] and not groups[1]&groups[2]
        for seed in cfg['seeds']:
            a=CAMERA/f'fold_{f["fold"]}/seed_{seed}/full/camera.pt'
            b=BASELINE/f'camera/fold_{f["fold"]}/seed_{seed}/full/camera.pt'
            assert sha(a)==sha(b)
    idx=np.array([0,1,296]);features,rotations=camera_features(data,0,17)
    changed={k:v.copy() for k,v in data.items()}
    for k in ['xy','boxes','H','R','cam','seconds']:changed[k]=changed[k][:,:P]
    cf,cr=camera_features(changed,0,17);assert np.array_equal(features,cf) and np.array_equal(rotations,cr)
    checks=[]
    for variant in VARIANTS:
        seed_all(17);model=Predictor(variant,cfg['capacity'][variant]['ff_width']).to(DEVICE);model.eval()
        h=history(data,idx,model.geometry);z=forward(model,h,features,idx);z.square().mean().backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        a=infer(model,data,idx,features,rotations);b=infer(model,changed,idx,cf,cr)
        assert np.array_equal(a['pred'],b['pred'])
        assert parameter_count(model)==cfg['capacity'][variant]['trainable_parameters']
        checks.append(dict(variant=variant,connected=True,future_suffix_removed_difference=0))
    for p,h in read(OUT/'locked_hashes.json').items():assert sha(ROOT/p)==h,p
    write(OUT/'preflight.json',dict(passed=True,checks=checks,exact_previous_train_lists=True,
        exact_previous_arrays=True,exact_validation_test=True,exact33_cameras=True,models=198))
    print('Preflight passed',flush=True)

if __name__=='__main__':main()
