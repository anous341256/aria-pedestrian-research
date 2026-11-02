"""Frozen-person 2x2 camera-condition/projection substitution. No training."""
from shared import *
from model import DualFlow, make_features
from train import sample_result, DEVICE, torch


def projected(component,camera,data,indices):
    pairs=[project_future(component[j],camera[i],data['K'][i]) for j,i in enumerate(indices)]
    return np.stack([p[0] for p in pairs]),np.stack([p[1] for p in pairs])


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    assert not (OUT/'complete.json').exists(), 'Preserve completed experiment'
    cfg=read(OLD/'protocol.json');folds=read(OLD/'folds.json');camfolds=read(CAMERA/'folds.json')
    meta=read(OLD/'samples.json');data=dict(np.load(OLD/'dataset.npz'))
    protocol=dict(date='2026-10-02',purpose='Synchronous replacement of predicted future camera conditioning and final projection',
                  training=False,frozen_person_variants=VARIANTS,seeds=cfg['seeds'],arms=ARMS,
                  arm_definition={'old_old':['old condition','old projection'],'old_new':['old condition','new projection'],
                                  'new_old':['new condition','old projection'],'new_new':['new condition','new projection']},
                  input_mode='Same reviewed historical SAM reference tracks and historical MPS as V1',
                  camera_source='Matching fold/seed full camera CV checkpoint; all-recording refits prohibited',
                  matched_noise='Exactly V1 five fixed sample starts, ten flow steps, mean trajectory',
                  evaluation='Same 117 windows, six recording-disjoint test folds, three seeds. No new sample selection or parameter fitting.',
                  invalid_projection_penalty_px=32000.,reference_camera='Diagnostic only; never supplied as inference conditioning',
                  visualization='Same nine fixed cases as V1; fixed seed 17. Additional extrema are explicitly post-score selections.')
    write(OUT/'protocol.json',protocol)
    sourcepaths=[OLD/'dataset.npz',OLD/'folds.json',OLD/'protocol.json',OLD/'samples.json',CAMERA/'folds.json',CAMERA/'protocol.json',
                 V1CODE/'model.py',V1CODE/'train.py',V1CODE/'common.py']
    for fold in folds:
        for seed in cfg['seeds']:
            sourcepaths.append(OLD/f'fold_{fold["fold"]}'/f'seed_{seed}'/'camera_predictions.npy')
            sourcepaths.append(CAMERA/f'fold_{fold["fold"]}'/f'seed_{seed}'/'full/predictions.npz')
            for variant in VARIANTS:
                sourcepaths.extend([OLD/f'fold_{fold["fold"]}'/f'seed_{seed}'/variant/name for name in ['checkpoint.pt','predictions_reviewed.npz']])
    hashes={str(p.relative_to(ROOT)):sha(p) for p in sourcepaths};write(OUT/'source_hashes.json',hashes)
    check=dict(baseline_prediction_max_difference=0.,baseline_component_max_difference=0.,future_suffix_prediction_max_difference=0.,
               camera_condition_uses_matching_new_predictions=True,history_inference_calls=0.,all_recording_refits_used=False,recording_split_valid=True)
    rows=[];count=0
    for fold in folds:
        f=fold['fold'];indices=np.array(fold['test_indices']);camerafold=camfolds[f]
        assert camerafold['test']==fold['test'] and camerafold['validation']==fold['validation']
        assert fold['test'] not in camerafold['train']
        for seed in cfg['seeds']:
            oldcam=np.load(OLD/f'fold_{f}'/f'seed_{seed}/camera_predictions.npy')
            newcam=np.load(CAMERA/f'fold_{f}'/f'seed_{seed}/full/predictions.npz')['person_rotation']
            for variant in VARIANTS:
                path=OLD/f'fold_{f}'/f'seed_{seed}'/variant
                model=DualFlow(cfg['hidden']).to(DEVICE)
                checkpoint=torch.load(path/'checkpoint.pt',map_location=DEVICE,weights_only=False)
                model.load_state_dict(checkpoint['model']);model.eval();model.requires_grad_(False)
                modelhash=sha(path/'checkpoint.pt')
                def forbid(*args):raise AssertionError('History head entered future inference')
                handle=model.history.register_forward_pre_hook(forbid)
                # Future targets may exist for scoring in the source helper, but never enter sample().
                features=make_features(data,indices,True,newcam,'reviewed')
                assert np.array_equal(features['camera'][:,P*3:],newcam[indices].reshape(len(indices),-1))
                old=sample_result(model,data,meta,indices,True,oldcam,'reviewed',cfg)
                new=sample_result(model,data,meta,indices,True,newcam,'reviewed',cfg)
                saved=np.load(path/'predictions_reviewed.npz');assert np.array_equal(saved['indices'],indices)
                diff=float(np.max(abs(old['pred']-saved['pred'])));cdiff=float(np.max(abs(old['component']-saved['component'])))
                check['baseline_prediction_max_difference']=max(check['baseline_prediction_max_difference'],diff)
                check['baseline_component_max_difference']=max(check['baseline_component_max_difference'],cdiff)
                assert diff<1e-4 and cdiff<1e-4,(f,seed,variant,diff,cdiff)
                if f==0 and seed==17:
                    changed={k:v.copy() for k,v in data.items()}
                    changed['xy'][:,P:]=12345.;changed['cam'][:,P:]=.5
                    changed['H'][:,P:]=np.eye(3);changed['R'][:,P:]=np.eye(3)
                    again=sample_result(model,changed,meta,indices,True,newcam,'reviewed',cfg)
                    d=float(np.max(abs(again['pred']-new['pred'])));assert d==0.
                    check['future_suffix_prediction_max_difference']=max(check['future_suffix_prediction_max_difference'],d)
                handle.remove()
                arrays=dict(indices=indices,old_component=old['component'],new_component=new['component'],
                            old_reference_camera=old['oracle'],new_reference_camera=new['oracle'],
                            old_samples=old['samples_normalized'],new_samples=new['samples_normalized'])
                for arm,component,cam in [('old_old',old['component'],oldcam),('old_new',old['component'],newcam),
                                           ('new_old',new['component'],oldcam),('new_new',new['component'],newcam)]:
                    pred,depth=projected(component,cam,data,indices);arrays[arm]=pred;arrays[arm+'_depth']=depth
                    for j,i in enumerate(indices):
                        rows.append(dict(index=int(i),key=meta[i]['key'],recording=meta[i]['recording'],clip=meta[i]['clip'],id=meta[i]['id'],
                                         seed=seed,fold=f,variant=variant,arm=arm,nonoverlap=meta[i]['nonoverlap'],edge=meta[i]['edge'],
                                         metrics=metrics(pred[j],data['xy'][i,P:],depth[j])))
                dest=OUT/f'fold_{f}'/f'seed_{seed}'/variant;dest.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(dest/'predictions.npz',**arrays)
                assert sha(path/'checkpoint.pt')==modelhash
                count+=1
        print('fold',f,'complete',flush=True)
    assert count==36
    for p,h in hashes.items():assert sha(ROOT/p)==h,p
    check.update(passed=True,frozen_person_models=count,source_files_unchanged=len(hashes),test_windows=len(meta))
    write(OUT/'metrics.json',rows);write(OUT/'verification.json',check);write(OUT/'complete.json',dict(passed=True,models=count,prediction_rows=len(rows)))
    print(check)


if __name__=='__main__':main()
