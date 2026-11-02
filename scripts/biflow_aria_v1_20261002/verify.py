"""Nontrivial contracts: time boundary, grouping, gradients and geometry."""
from train import *

def main():
    cfg=read(OUT/'protocol.json');data=dict(np.load(OUT/'dataset.npz'));meta=read(OUT/'samples.json');folds=read(OUT/'folds.json');checks={}
    assigned=[]
    for f in folds:
        a,b,c=set(f['train']),{f['validation']},{f['test']};assert not(a&b or b&c or a&c)
        assert all(meta[i]['recording'] in a for i in f['train_indices']);assert all(meta[i]['recording'] in b for i in f['val_indices']);assert all(meta[i]['recording'] in c for i in f['test_indices']);assigned+=f['test_indices']
    assert sorted(assigned)==list(range(len(meta)));checks['each_window_tested_once_per_seed']=True
    # Matched capacity: loss switch never changes network dimensions.
    model=DualFlow().to(DEVICE);checks['total_parameters']=sum(p.numel() for p in model.parameters());checks['inference_parameters']=sum(p.numel() for n,p in model.named_parameters() if not n.startswith('history.'))
    idx=np.array([0,20,80]);cam=np.zeros((len(meta),F,3),np.float32)
    errors=[]
    for comp in [False,True]:
        a=make_features(data,idx,comp,cam);changed={k:v.copy() for k,v in data.items()};changed['xy'][:,P:]+=1e5;changed['cam'][:,P:]+=5;changed['H'][:,P:]*=2
        b=make_features(changed,idx,comp,cam)
        for key in ['seq','origin','camera']:assert np.array_equal(a[key],b[key]),key
        model.eval();ta,tb=tensors(a),tensors(b);noise=tensor(noise_for(meta,idx,'future'))
        pa=model.sample(ta['seq'],ta['origin'],ta['camera'],noise);pb=model.sample(tb['seq'],tb['origin'],tb['camera'],noise);err=float((pa-pb).abs().max());assert err==0.;errors.append(err)
        # Inference must not call the historical decoder.
        def forbidden(*args):raise AssertionError('History decoder entered future inference')
        hook=model.history.register_forward_pre_hook(forbidden)
        model.sample(ta['seq'],ta['origin'],ta['camera'],noise);hook.remove()
    checks['future_suffix_mutation_prediction_max_difference']=errors;checks['future_inference_does_not_execute_history_head']=True
    # Reconstruction must train the same encoder, without future targets as inputs.
    a=tensors(make_features(data,idx,True,cam,'controlled_noise',np.random.default_rng(5)));model.train();h=model.encode(a['seq'],a['origin'],a['camera']);x=torch.randn(len(idx),P*2,device=DEVICE)*.15;t=torch.zeros(len(idx),device=DEVICE);loss=(model.history(x,t,h)-a['past']).square().mean();model.zero_grad();loss.backward();g=sum(p.grad.abs().sum().item() for p in model.encoder.parameters() if p.grad is not None);assert g>0;checks['reconstruction_gradient_reaches_shared_encoder']=g
    # Rotation cancellation and re-composition: reference XY recovered without a predictor.
    errs=[]
    for i in range(len(meta)):
        normalized,_=warp(data['xy'][i],data['H'][i]);back,z=project_future(normalized[P:],data['cam'][i,P:],data['K'][i]);errs.append(float(np.max(np.abs(back-data['xy'][i,P:]))))
    assert max(errs)<.01;checks['max_rotation_roundtrip_pixel_error']=max(errs)
    for p,digest in read(OUT/'sources.json').items():assert sha(ROOT/p.replace('\\','/'))==digest,p
    checks['source_annotations_and_videos_unchanged']=True;checks['passed']=True;write(OUT/'verification.json',checks);print(checks)
if __name__=='__main__':main()
