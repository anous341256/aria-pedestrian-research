from shared import *
from network import Predictor,choose_width,parameter_count,torch
from run import camera_features,infer,forward
from train import tensor,seed_all,DEVICE


def main():
    assert not (OUT/'protocol.json').exists(),'Protocol already fixed'
    folds=read(OLD/'folds.json');cfolds=read(CAMERA/'folds.json');meta=read(OLD/'samples.json')
    for f,c in zip(folds,cfolds):
        assert f['test']==c['test'] and f['validation']==c['validation']
        assert not set(c['train'])&{f['test'],f['validation']}
        sets=[set(meta[i]['recording'] for i in f[k]) for k in ['train_indices','val_indices','test_indices']]
        assert not sets[0]&sets[1] and not sets[0]&sets[2] and not sets[1]&sets[2]
    assert sorted(i for f in folds for i in f['test_indices'])==list(range(117))
    capacity={v:choose_width(v) for v in VARIANTS}
    for v,c in capacity.items():
        c['frozen_active_parameters']=8589 if v[0] in 'CF' else (3552 if v[0] in 'ADE' else 0)
        c['total_active_parameters']=c['trainable_parameters']+c['frozen_active_parameters']
    counts=[x['trainable_parameters'] for x in capacity.values()]
    assert (max(counts)-min(counts))/np.mean(counts)<.005
    cfg=dict(name='Six Transformer fusion comparisons',approved='User confirmed execution 2026-10-05',variants=VARIANTS,
             seeds=[17,29,43],epochs=120,updates_per_epoch=2,batch_size=48,validation_every=10,
             learning_rate=.001,weight_decay=.01,gradient_clip=1.,hidden=64,heads=4,dropout=0.,capacity=capacity,
             evaluation_inputs=['reviewed','pre_review','controlled_noise'],
             selection='Validation final image ADE for all arms',loss='Normalized future MSE in arm output coordinates; no reconstruction',
             comparison='A/B/D/E raw output; C/F residual output+projection. Different output families are system comparisons.',
             camera='Frozen matching full-data recording-held-out GRU encoder; C/F same independent full forecast',
             plan='plans/TRANSFORMER_FUSION_20261005.md',invalid_projection_penalty_px=32000.,
             condition_bins=dict(final_rotation_degrees=[5,15],box_height_fraction=[.15,.3]),
             primary='Report all six predeclared arms; lowest test mean is exploratory ranking, not validation-selected deployment',
             labels='Offline reviewed SAM2 references and historical MPS; not independently validated GT or RGB-only')
    write(OUT/'protocol.json',cfg);write(OUT/'folds.json',folds)
    paths=[OLD/x for x in ['dataset.npz','samples.json','folds.json','protocol.json']]
    paths += [ROOT/'plans/TRANSFORMER_FUSION_20261005.md',CAMERA/'folds.json']
    paths += list((ROOT/'scripts/biflow_aria_v1_20261002').glob('*.py'))
    paths += list((ROOT/'models/forecasting/biflow_comp_dual_baseline_20261002').rglob('*'))
    for base in [OLD,PREVIOUS,CAMERA]:
        paths += list(base.glob('fold_*/seed_*/*/*.pt'))
        paths += list(base.glob('fold_*/seed_*/*/predictions*.npz'))
    for cid in sorted({m['clip'] for m in meta}):
        paths += [ROOT/f'outputs/sam2_near_people_20261002/{cid}/final/masks_rle.jsonl.gz',
                  ROOT/f'outputs/sam2_g1_v6_v8_20261001/{cid}/input/rgb.mp4']
    paths=sorted(set(p for p in paths if p.is_file()))
    write(OUT/'source_hashes.json',{p.relative_to(ROOT).as_posix():sha(p) for p in paths})
    data=dict(np.load(OLD/'dataset.npz'));features,rotations=camera_features(data,0,17);idx=np.array([0,1,2])
    changed={k:v.copy() for k,v in data.items()}
    for k in ['xy','boxes','H','R','cam','seconds']:changed[k][:,P:]=np.random.default_rng(10).normal(size=changed[k][:,P:].shape)
    cf,cr=camera_features(changed,0,17);assert np.array_equal(cf,features) and np.array_equal(cr,rotations)
    checks=[]
    for v in VARIANTS:
        seed_all(17);m=Predictor(v,capacity[v]['ff_width']).to(DEVICE);m.eval()
        h=history(data,idx,m.geometry);z=forward(m,h,features,idx)
        z.square().mean().backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
        a=infer(m,data,idx,features,rotations);b=infer(m,changed,idx,cf,cr)
        assert np.array_equal(a['pred'],b['pred'])
        if not m.uses_camera_features:
            other=infer(m,data,idx,features+100,rotations)
            assert np.array_equal(a['component'],other['component'])
        checks.append(dict(variant=v,parameters=parameter_count(m),all_parameters_connected=True,future_mutation_max=0.))
    write(OUT/'preflight.json',dict(passed=True,checks=checks,source_files=len(paths),split_recording_disjoint=True,camera_future_mutation_max=0.))
    print('Preflight passed; capacity:',capacity,flush=True)


if __name__=='__main__':main()
