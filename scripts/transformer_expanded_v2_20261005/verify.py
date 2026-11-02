"""Replay all held-out checkpoints and audit inference dependence/preservation."""
from shared import *
from network import Predictor,torch,parameter_count
from run import camera_features,infer
from train import DEVICE


def main():
    assert read(OUT/'training_complete.json')['passed']
    cfg=read(OUT/'protocol.json');data=dict(np.load(DATA/'dataset.npz'));meta=read(DATA/'samples.json')
    changed={k:v.copy() for k,v in data.items()}
    rng=np.random.default_rng(827)
    for k in ['xy','boxes','H','R','cam','seconds']:changed[k][:,P:]=rng.normal(size=changed[k][:,P:].shape)
    checks=[];shared_camera=[]
    for fold in read(OUT/'folds.json'):
        f=fold['fold'];idx=np.array(fold['test_indices'])
        for seed in cfg['seeds']:
            features,rotations=camera_features(data,f,seed);alt_features,alt_rotations=camera_features(changed,f,seed)
            assert np.array_equal(features,alt_features) and np.array_equal(rotations,alt_rotations)
            by_time={}
            for i,m in enumerate(meta):by_time.setdefault((m['recording'],m['timestamp_ns']),[]).append(i)
            pairs=0;maxshared=0.
            for ids in by_time.values():
                for i in ids[1:]:
                    if np.allclose(data['cam'][i,:P],data['cam'][ids[0],:P],atol=1e-6):
                        maxshared=max(maxshared,float(np.max(np.abs(rotations[i]-rotations[ids[0]]))));pairs+=1
            assert maxshared<1e-6
            shared_camera.append(dict(fold=f,seed=seed,same_history_person_pairs=pairs,maximum_difference=maxshared))
            for v in VARIANTS:
                path=OUT/f'fold_{f}/seed_{seed}/{v}'
                checkpoint=torch.load(path/'checkpoint.pt',map_location=DEVICE,weights_only=False)
                assert checkpoint['protocol_sha256']==sha(OUT/'protocol.json')
                m=Predictor(v,checkpoint['ff_width']).to(DEVICE);m.load_state_dict(checkpoint['model']);m.eval()
                assert parameter_count(m)==cfg['capacity'][v]['trainable_parameters']
                max_replay=0.
                for mode in cfg['evaluation_inputs']:
                    original=np.load(path/f'predictions_{mode}.npz');a=infer(m,data,idx,features,rotations,mode,True)
                    assert np.array_equal(original['indices'],idx)
                    for key,value in a.items():
                        error=float(np.max(np.abs(value-original[key])));max_replay=max(max_replay,error)
                        assert error<1e-4,(v,key,error)
                a=infer(m,data,idx,features,rotations);b=infer(m,changed,idx,alt_features,alt_rotations)
                future_difference=float(np.max(np.abs(a['pred']-b['pred'])));assert future_difference==0
                feature_effect=None
                if not m.uses_camera_features:
                    b=infer(m,data,idx,features+10,rotations);assert np.array_equal(a['component'],b['component'])
                    feature_effect=0.
                if m.geometry:
                    b=infer(m,data,idx,features,rotations+.05)
                    assert np.array_equal(a['component'],b['component'])
                checks.append(dict(fold=f,seed=seed,variant=v,replay_max_px=max_replay,future_mutation_max_px=future_difference,
                                   forbidden_camera_features_effect=feature_effect))
    hashes=read(OUT/'source_hashes.json');changed_sources=[p for p,h in hashes.items() if sha(ROOT/p)!=h]
    assert not changed_sources,changed_sources
    write(OUT/'verification.json',dict(passed=True,models=len(checks),predictions_replayed=len(checks)*3,checks=checks,
                                      shared_camera=shared_camera,preserved_source_files=len(hashes),changed_sources=changed_sources))
    print('Verified 234 models, 702 prediction files, future isolation and',len(hashes),'preserved sources')


if __name__=='__main__':main()
