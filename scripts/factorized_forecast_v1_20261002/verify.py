"""Check causal interfaces, branch separation, geometry, and frozen camera folds."""
from run import *
from model import CameraPredictor


def main():
    cfg=read(OUT/'protocol.json'); data=dict(np.load(OLD/'dataset.npz')); meta=read(OLD/'samples.json')
    folds=read(OUT/'folds.json'); cf=read(CAMERA/'folds.json'); assigned=[]; check={}
    for fold in folds:
        train=set(fold['train']); val={fold['validation']}; test={fold['test']}
        assert not (train&val or train&test or val&test)
        for name,allowed in [('train',train),('val',val),('test',test)]:
            assert all(meta[i]['recording'] in allowed for i in fold[name+'_indices'])
        c=cf[fold['fold']]
        assert c['test']==fold['test'] and c['validation']==fold['validation']
        assert not ((test|val)&set(c['train'])) and train.issubset(c['train'])
        assigned+=fold['test_indices']
    assert sorted(assigned)==list(range(len(meta))); check['recording_split_and_camera_split_valid']=True
    idx=np.array([0,20,80]); cam=cameras(0,17)['full']; h=input_history(data,idx)
    changed={k:v.copy() for k,v in data.items()}
    changed['xy'][:,P:]=12345.; changed['boxes'][:]=999.; changed['cam'][:,P:]=.5
    changed['R'][:,P:]=np.eye(3); changed['H'][:,P:]=np.eye(3)
    changed['seconds'][:,P:]=12345.; changed['camera'][:]=7.
    altered=input_history(changed,idx)
    assert all(np.array_equal(h[k],altered[k]) for k in h)
    check['future_suffix_and_raw_box_mutation_input_difference']=0.
    for coupled in [False,True]:
        seed_all(17); model=PedestrianPredictor(coupled).to(DEVICE).eval()
        a=infer_component(model,data,meta,idx,cam,cfg)
        b=infer_component(model,changed,meta,idx,cam,cfg)
        diff=float(np.max(abs(a['component']-b['component']))); assert diff==0.
        check['coupled' if coupled else 'separated']={'parameters':sum(p.numel() for p in model.parameters()),'future_suffix_prediction_difference':diff}
        # Changing only predicted camera must leave separated residual output exactly fixed.
        b=infer_component(model,data,meta,idx,cam+.3,cfg)
        cdiff=float(np.max(abs(a['component']-b['component'])))
        if not coupled:
            assert cdiff==0.
            final1,_=project(a['component'],cam,data,idx); final2,_=project(a['component'],cam+.3,data,idx)
            assert np.max(abs(final1-final2))>1.
            check['separated_camera_condition_difference']=cdiff
            check['separated_final_projection_responds_to_camera']=True
            try: model.encode(tensor(h['seq']),tensor(h['origin']),tensor(camera_condition(data,idx,cam)))
            except ValueError: pass
            else: raise AssertionError('Camera accepted in separated pedestrian module')
        else: assert cdiff>0.; check['coupled_camera_condition_response_px']=cdiff
        def forbid(*args): raise AssertionError('History decoder entered future inference')
        handle=model.history.register_forward_pre_hook(forbid)
        infer_component(model,data,meta,idx,cam,cfg); handle.remove()
    check['future_inference_history_head_not_called']=True
    model=PedestrianPredictor().to(DEVICE); args=model_inputs(model,h,data,idx,cam)
    target=targets(data,idx,h['origin']); latent=model.encode(**args)
    loss=(model.history(tensor(np.zeros((len(idx),P*2))),tensor(np.zeros(len(idx))),latent)-tensor(target['past'])).square().mean()
    loss.backward(); grad=sum(p.grad.abs().sum().item() for p in model.encoder.parameters() if p.grad is not None)
    assert grad>0.; check['history_reconstruction_encoder_gradient']=grad
    errors=[]
    for i in range(len(meta)):
        stable=warp(data['xy'][i],data['H'][i])[0]
        back,_=project_future(stable[P:],data['cam'][i,P:],data['K'][i]); errors.append(float(np.max(abs(back-data['xy'][i,P:]))))
    assert max(errors)<.01; check['geometry_roundtrip_max_px']=max(errors)
    # Change historical image coordinates and their matching rotation homographies.
    synthetic={k:v.copy() for k,v in data.items()}; rng=np.random.default_rng(51)
    for i in idx:
        for j in range(P-1):
            rotation=rotmat([rng.normal(0,.03,3)])[0]; K=data['K'][i].astype(float)
            A=K@rotation@np.linalg.inv(K)
            synthetic['xy'][i,j]=warp(data['xy'][i,j],A)[0]
            synthetic['H'][i,j]=data['H'][i,j]@np.linalg.inv(A)
    invariant=input_history(synthetic,idx)
    d=float(np.max(abs(invariant['seq']-h['seq']))); assert d<1e-5
    check['synthetic_rotation_stable_feature_max_difference']=d
    replay=0.
    for fold in folds:
        for seed in cfg['seeds']:
            camera=CameraPredictor().to(DEVICE).eval()
            path=CAMERA/f'fold_{fold["fold"]}/seed_{seed}/full/camera.pt'
            camera.load_state_dict(torch.load(path,map_location=DEVICE,weights_only=False)['model'])
            with torch.no_grad(): prediction=camera(tensor(data['cam'][:,:P])).cpu().numpy()
            cached=cameras(fold['fold'],seed)['full']; delta=float(np.max(abs(prediction-cached)))
            assert delta<1e-5; replay=max(replay,delta)
    check['camera_replay_max_rotvec_difference']=replay
    for p,digest in read(OUT/'source_hashes.json').items(): assert sha(ROOT/p)==digest,p
    check.update(passed=True,full_data_camera_models_verified=18,all_recording_refits_used=False)
    write(OUT/'preflight.json',check); print(check)


if __name__=='__main__': main()
