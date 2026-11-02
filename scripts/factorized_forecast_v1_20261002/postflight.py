"""Replay trained weights and verify actual branch independence after training."""
from run import *


def main():
    cfg=read(OUT/'protocol.json');data=dict(np.load(OLD/'dataset.npz'));meta=read(OLD/'samples.json')
    replay=0.;future=0.;separation=0.;count=0;coupled_response=[];camera_shared_max=0.
    for fold in read(OUT/'folds.json'):
        idx=np.array(fold['test_indices'])
        changed={k:v.copy() for k,v in data.items()}
        changed['xy'][:,P:]=12345.;changed['cam'][:,P:]=.6;changed['H'][:,P:]=np.eye(3)
        changed['R'][:,P:]=np.eye(3);changed['boxes'][:]=999.
        for seed in cfg['seeds']:
            cam=cameras(fold['fold'],seed)['full']
            groups={}
            for i in idx:groups.setdefault((meta[i]['recording'],meta[i]['timestamp_ns']),[]).append(i)
            for indices in groups.values():
                camera_shared_max=max(camera_shared_max,float(np.max(abs(cam[indices]-cam[indices[0]]))))
            for variant in VARIANTS:
                dest=OUT/f'fold_{fold["fold"]}/seed_{seed}/{variant}'
                model=PedestrianPredictor(variant.startswith('coupled')).to(DEVICE).eval()
                checkpoint=torch.load(dest/'checkpoint.pt',map_location=DEVICE,weights_only=False)
                assert checkpoint['fold']==fold and checkpoint['seed']==seed
                assert checkpoint['protocol_sha256']==sha(OUT/'protocol.json')
                model.load_state_dict(checkpoint['model']);model.requires_grad_(False)
                def forbid(*args):raise AssertionError('History head called by future inference')
                handle=model.history.register_forward_pre_hook(forbid)
                a=infer_component(model,data,meta,idx,cam,cfg)
                z=np.load(dest/'predictions_reviewed.npz')
                d=float(np.max(abs(a['component']-z['component'])));assert d<1e-4;replay=max(replay,d)
                b=infer_component(model,changed,meta,idx,cam,cfg)
                d=float(np.max(abs(a['component']-b['component'])));assert d==0.;future=max(future,d)
                b=infer_component(model,data,meta,idx,cam+.1,cfg)
                d=float(np.max(abs(a['component']-b['component'])))
                if model.coupled:coupled_response.append(d)
                else:assert d==0.;separation=max(separation,d)
                handle.remove();count+=1
    assert count==54 and camera_shared_max<1e-5
    for p,h in read(OUT/'source_hashes.json').items():assert sha(ROOT/p)==h,p
    write(OUT/'postflight.json',dict(passed=True,trained_models_replayed=count,prediction_replay_max_px=replay,
        future_suffix_prediction_difference=future,separated_camera_mutation_component_difference=separation,
        coupled_camera_mutation_component_max_px=max(coupled_response),same_timestamp_camera_sharing_difference=camera_shared_max,
        all_recording_refits_used=False,source_files_unchanged=len(read(OUT/'source_hashes.json'))))
    print(read(OUT/'postflight.json'))


if __name__=='__main__':main()
