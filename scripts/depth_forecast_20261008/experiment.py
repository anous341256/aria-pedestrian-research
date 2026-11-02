"""Paired no-depth/depth continuation on expanded data, fixed original117 tests."""
import sys,json,copy,time,hashlib,csv,importlib.util,argparse
from pathlib import Path
import numpy as np
from depth_model import DepthAttention,torch
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/depth_forecast_20261008';TRAIN=OUT/'forecast'
sys.path.insert(0,str(ROOT/'scripts/dual_cross_attention_20261008'))
spec=importlib.util.spec_from_file_location('previous_cross',ROOT/'scripts/dual_cross_attention_20261008/experiment.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
ARMS=['no_depth','with_depth'];SEEDS=[17,29,43]
read=old.read;write=old.write

def load(fold,seed,arm):
    base=old.load(fold,seed,True)
    ck=torch.load(old.OUT/f'fold_{fold}/seed_{seed}/cross_attention/checkpoint.pt',map_location='cpu',weights_only=False)
    old.seed_all(seed);model=DepthAttention(base.camera,base.person,arm=='with_depth').to(old.DEVICE)
    missing,unexpected=model.load_state_dict(ck['model'],strict=False)
    assert missing==['depth_embed.weight'] and not unexpected
    return model

def depth_features(raw):
    # Raw channels [Z,IQR,LR fraction,point count,usable]. Access only history.
    h=raw[:,:6];valid=h[:,:,4]>0;z=np.maximum(h[:,:,0],.5)
    logz=np.log(z);delta=np.zeros_like(logz);delta[:,1:]=np.clip(np.diff(logz,axis=1),-1,1)*(valid[:,1:]&valid[:,:-1])
    result=np.stack([logz/np.log(30),delta,valid.astype(float),h[:,:,2],np.clip(h[:,:,1]/z,0,1)],-1)
    result[~valid]=0
    return result.astype(np.float32)

def args(data,meta,idx,mode='reviewed',rng=None):
    a,h=old.args(data,meta,idx,mode,rng);dep=depth_features(data['stereo'][idx])
    dep*=h['input_valid'][...,None];a['depth']=old.tensor(dep)
    return a,h

@torch.no_grad()
def predict(model,data,meta,idx):
    a,h=args(data,meta,idx);p,r=model.predict(**a,noise=old.tensor(old.noise_for(meta,idx,'future',5)))
    return dict(component=(p.cpu().numpy()+h['origin'][:,None])*old.SCALE,rotation=r.cpu().numpy())

def scales(model,data,meta,idx):
    z=old.diagnostics(predict(model,data,meta,idx),data,idx);truth=old.warp(data['xy'][idx,6:],data['H'][idx,6:])[0]
    groups=old.group_ids(meta,idx).cpu().numpy();unique=[np.flatnonzero(groups==g)[0] for g in np.unique(groups)]
    return dict(person_mse=max(1e-5,float(np.mean(((z['component']-truth)/old.SCALE)**2))),camera_mse=max(1e-4,float(np.mean((z['rotation'][unique]-data['cam'][idx,6:][unique])**2))),residual_ADE=max(10.,float(z['residual_error'].mean())),camera_angle=max(1.,float(z['angle_deg'][unique].mean())))

def package(history_only=False):
    plan=read(OUT/'extraction_plan.json');raw=np.zeros((374,21,5),np.float32);missing=[];cache={}
    for m in plan['sample_refs']:
        for j,(key,target) in enumerate(zip(m['frame_keys'],m['target_keys'])):
            if history_only and j>=6:continue
            path=OUT/'frames'/key/'measurements.json'
            if not path.exists():
                if (OUT/'frames'/key/'failure.json').exists():missing.append(dict(sample=m['key'],frame=j,reason='extraction_failed'));continue
                raise RuntimeError('Extraction not yet complete: '+key)
            if key not in cache:cache[key]={r['target_key']:r for r in read(path)}
            r=cache[key][target];raw[m['index'],j]=[r['z_median_m'] or 0,r['z_iqr_m'] or 0,r['lr_fraction'],r['points'],r['usable']]
    path=OUT/('depth_history.npz' if history_only else 'depth_all21.npz');np.savez_compressed(path,depth=raw)
    write(OUT/('depth_history_summary.json' if history_only else 'depth_all21_summary.json'),dict(shape=list(raw.shape),historical_usable_frames=int(raw[:,:6,4].sum()),total_historical_frames=374*6,windows_any_history=int((raw[:,:6,4].sum(1)>0).sum()),windows_all6_history=int((raw[:,:6,4].sum(1)==6).sum()),failed_frames=missing))
    return raw

def prepare():
    TRAIN.mkdir(parents=True,exist_ok=True);plan=read(OUT/'extraction_plan.json');data=dict(np.load(ROOT/'outputs/transformer_uncovered200_20261006/data/dataset.npz'));meta=read(ROOT/'outputs/transformer_uncovered200_20261006/data/samples.json')
    data['stereo']=package(history_only=True);folds=copy.deepcopy(read(old.OUT/'folds.json'));union=plan['actual_previous_train_union']
    protected=read(OUT/'protected_sources.json')
    for folder in ['dual_cross_attention_20261008','factorized_forecast_v1_20261002','biflow_aria_v1_20261002']:
        for p in (ROOT/'scripts'/folder).glob('*.py'):protected[str(p.relative_to(ROOT))]=old.sha(p)
    code_hashes={p.name:old.sha(p) for p in [Path(__file__),Path(__file__).with_name('depth_model.py')]}
    if (TRAIN/'training_code_hashes.json').exists():assert read(TRAIN/'training_code_hashes.json')==code_hashes
    else:write(TRAIN/'training_code_hashes.json',code_hashes)
    for f in folds:
        f['previous_train_indices']=f['train_indices'];f['train_indices']=[i for i in union if meta[i]['recording'] not in [f['test'],f['validation']]]
        f['train']=sorted({meta[i]['recording'] for i in f['train_indices']})
        assert set(f['previous_train_indices'])<=set(f['train_indices'])
        assert not {meta[i]['recording'] for i in f['train_indices']} & {f['test'],f['validation']}
        for seed in SEEDS:
            for p in [old.OUT/f'fold_{f["fold"]}/seed_{seed}/cross_attention/checkpoint.pt',old.CAM/f'fold_{f["fold"]}/seed_{seed}/full/camera.pt',old.BASE/f'fold_{f["fold"]}/seed_{seed}/separated_dual/checkpoint.pt']:
                protected[str(p.relative_to(ROOT))]=old.sha(p)
    cfg=dict(arms=ARMS,seeds=SEEDS,updates=240,validation_every=20,lr=.001,weight_decay=.01,anchor_batch=48,history_weight=.5,test_windows=117,folds=6,cohort=374,depth='historical metric stereo surface depth plus validity/quality; full zeroing control',initial='previous selected cross checkpoint; zero depth embedding; frozen original backbones',selection='same normalized residual and camera validation objective; step0 eligible',no_test_tuning=True)
    if (TRAIN/'protocol.json').exists():assert read(TRAIN/'protocol.json')==cfg and read(TRAIN/'folds.json')==folds
    else:write(TRAIN/'protocol.json',cfg);write(TRAIN/'folds.json',folds);write(TRAIN/'protected_sources.json',protected)
    return data,meta,folds,cfg

def signature(model):return hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in model.state_dict().values())).hexdigest()

def fit(data,meta,f,seed,arm,cfg):
    dest=TRAIN/f'fold_{f["fold"]}/seed_{seed}/{arm}';dest.mkdir(parents=True,exist_ok=True)
    if (dest/'complete.json').exists():return
    begin=time.monotonic();model=load(f['fold'],seed,arm).eval();initial=signature(model)
    scale=scales(model,data,meta,np.array(f['train_indices']));vi=np.array(f['val_indices']);z=old.diagnostics(predict(model,data,meta,vi),data,vi)
    best=old.score(z,meta,vi,scale);state=copy.deepcopy(model.state_dict());beststep=0;log=[dict(step=0,score=best)]
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.001,weight_decay=.01);draws=hashlib.sha256()
    for step in range(1,241):
        model.train();rng=np.random.default_rng(seed*100000+step);idx=old.batch(f['train_indices'],meta,rng);a,h=args(data,meta,idx,'train',rng);y=old.targets(data,idx,h['origin']);n=len(idx)
        t1=rng.random(n);t2=rng.random(n);n1=rng.normal(0,.15,(n,12));n2=rng.normal(0,.15,(n,30))
        for val in [idx,h['seq'],t1,t2,n1,n2]:draws.update(val.tobytes())
        loss,parts=model.loss(**a,past=old.tensor(y['past']),future=old.tensor(y['future']),rotation=old.tensor(data['cam'][idx,6:]),t1=old.tensor(t1),t2=old.tensor(t2),n1=old.tensor(n1),n2=old.tensor(n2),scales=scale)
        assert torch.isfinite(loss);opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
        if step%20==0:
            model.eval();z=old.diagnostics(predict(model,data,meta,vi),data,vi);value=old.score(z,meta,vi,scale);log.append(dict(step=step,score=value,loss=float(loss)))
            if value<best:best=value;beststep=step;state=copy.deepcopy(model.state_dict())
    model.load_state_dict(state);model.eval();idx=np.array(f['test_indices']);z=old.diagnostics(predict(model,data,meta,idx),data,idx)
    torch.save(dict(model=state,fold=f,seed=seed,arm=arm,step=beststep,scales=scale),dest/'checkpoint.pt');np.savez_compressed(dest/'predictions.npz',indices=idx,**z);write(dest/'training.json',log)
    write(dest/'complete.json',dict(seconds=time.monotonic()-begin,step=beststep,initial_hash=initial,draw_hash=draws.hexdigest(),parameters=sum(p.numel() for p in model.parameters()),trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)))
    print('FIT',f['fold'],seed,arm,beststep,flush=True)

def evaluate(data,meta,folds,cfg):
    rows=[];replays=0;maxgeo=0.;maxshared=0.;total=0
    short={k:(v[:,:6].copy() if k in ['xy','boxes','R','cam','H','seconds','stereo'] else v.copy()) for k,v in data.items()}
    for f in folds:
        idx=np.array(f['test_indices'])
        for seed in SEEDS:
            records=[]
            for arm in ARMS:
                dest=TRAIN/f'fold_{f["fold"]}/seed_{seed}/{arm}';ck=torch.load(dest/'checkpoint.pt',map_location=old.DEVICE,weights_only=False);model=load(f['fold'],seed,arm).eval();start=copy.deepcopy(model.state_dict());model.load_state_dict(ck['model'])
                for key,v in start.items():
                    if key.startswith(('camera.','person.')):assert torch.equal(v,model.state_dict()[key])
                actual=predict(model,data,meta,idx);truncated=predict(model,short,meta,idx);saved=dict(np.load(dest/'predictions.npz'))
                for k in actual:assert np.array_equal(actual[k],truncated[k]) and np.array_equal(actual[k],saved[k]),(f['fold'],seed,arm,k)
                if arm=='no_depth':
                    changed=dict(short,stereo=np.ones_like(short['stereo'])*10);test=predict(model,changed,meta,idx)
                    assert all(np.array_equal(actual[k],test[k]) for k in actual)
                z=old.diagnostics(actual,data,idx);group=old.group_ids(meta,idx).cpu().numpy()
                for g in np.unique(group):
                    same=z['rotation'][group==g];d=float(np.max(np.abs(same-same[0])));assert d<1e-6;maxshared=max(maxshared,d)
                for j,i in enumerate(idx):
                    rays=np.c_[z['component'][j],np.ones(15)]@np.linalg.inv(data['K'][i]).T
                    transformed=np.einsum('tji,tj->ti',old.quaternion_matrix(z['rotation'][j]),rays)@data['K'][i].T
                    xy=transformed[:,:2]/transformed[:,2:];maxgeo=max(maxgeo,float(np.max(np.abs(xy-z['image'][j]))));assert maxgeo<.01
                    r=dict(index=int(i),key=meta[i]['key'],recording=meta[i]['recording'],timestamp_ns=meta[i]['timestamp_ns'],seed=seed,arm=arm,nonoverlap=meta[i]['nonoverlap'],history_depth_valid=int(data['stereo'][i,:6,4].sum()))
                    for name,errors in [('image',z['image_error']),('residual',z['residual_error']),('camera_deg',z['angle_deg']),('gt_camera',z['gt_camera_error']),('gt_person',z['gt_person_error'])]:r[name+'_ADE']=float(errors[j].mean());r[name+'_FDE']=float(errors[j,-1])
                    rows.append(r)
                rec=read(dest/'complete.json');records.append(rec);total+=rec['seconds'];replays+=1
            assert records[0]['initial_hash']==records[1]['initial_hash'] and records[0]['draw_hash']==records[1]['draw_hash']
    metrics=[k for k in rows[0] if k.endswith(('_ADE','_FDE'))];summary={}
    for arm in ARMS:
        rr=[r for r in rows if r['arm']==arm];summary[arm]={k:float(np.mean([r[k] for r in rr])) for k in metrics}
        summary[arm]['per_seed']={str(s):{k:float(np.mean([r[k] for r in rr if r['seed']==s])) for k in metrics} for s in SEEDS}
        summary[arm]['recording_equal']={k:float(np.mean([np.mean([r[k] for r in rr if r['recording']==rec]) for rec in sorted({r['recording'] for r in rr})])) for k in metrics}
        summary[arm]['sparse']={k:float(np.mean([r[k] for r in rr if r['nonoverlap']])) for k in metrics}
    with (TRAIN/'metrics.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    hold=np.linalg.norm(data['xy'][:117,6:]-data['xy'][:117,5:6],axis=-1)
    write(TRAIN/'comparison.json',dict(arms=summary,hold=dict(ADE=float(hold.mean()),FDE=float(hold[:,-1].mean())),training_seconds=total))
    protected=read(TRAIN/'protected_sources.json');assert all(old.sha(ROOT/p)==v for p,v in protected.items())
    write(TRAIN/'verification.json',dict(passed=True,model_replays=replays,exact_future_prefix=True,no_depth_invariance=True,paired_initialization_draws=True,frozen_backbones=True,shared_camera_max_rad=maxshared,independent_projection_max_px=maxgeo,protected_sources_unchanged=len(protected)))
    print(json.dumps(summary,indent=2),flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['train','package','preflight']);a=ap.parse_args()
    if a.mode=='package':package(False);return
    data,meta,folds,cfg=prepare()
    if a.mode=='preflight':
        idx=np.array(folds[0]['test_indices']);m=load(0,17,'no_depth').eval();z=predict(m,data,meta,idx);base=old.load(0,17,True).eval();base.load_state_dict(torch.load(old.OUT/'fold_0/seed_17/cross_attention/checkpoint.pt',map_location=old.DEVICE,weights_only=False)['model']);b=old.predict(base,data,meta,idx)
        assert all(np.array_equal(z[k],b[k]) for k in z);print('INITIAL REPLAY PASSED');return
    for f in folds:
        for s in SEEDS:
            for arm in ARMS:fit(data,meta,f,s,arm,cfg)
    evaluate(data,meta,folds,cfg)

if __name__=='__main__':main()
