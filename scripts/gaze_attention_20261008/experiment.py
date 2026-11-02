"""Paired gaze input ablation, original117, exact recording-held-out folds."""
from pathlib import Path
import sys,time,copy,json,csv,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/transformer_fusion_20261005'))
from shared import history,target,read,write,sha,SCALE,P,F,OLD
sys.path.insert(0,str(Path(__file__).parent))
from network import Predictor,torch
from train import seed_all,balanced,tensor,DEVICE
OUT=ROOT/'outputs/gaze_attention_20261008'

def forward(model,data,gaze,idx,use_gaze,mode='reviewed',rng=None):
    h=history(data,idx,False,mode,rng)
    camera=np.concatenate([data['cam'][idx,:P],data['seconds'][idx,:P,None]/.5],axis=-1)
    offsets=model(tensor(h['seq']),tensor(camera),tensor(gaze['values'][idx]),
        torch.as_tensor(gaze['valid'][idx],device=DEVICE),tensor(h['condition']),use_gaze)
    return offsets,h

@torch.no_grad()
def infer(model,data,gaze,idx,use_gaze):
    offsets,h=forward(model,data,gaze,idx,use_gaze)
    return (offsets.cpu().numpy().reshape(len(idx),F,2)+h['origin'][:,None])*SCALE

def code_hashes():return {p.name:sha(p) for p in Path(__file__).parent.glob('*.py')}

def preflight(data,meta,gaze,folds):
    assert len(meta)==117 and data['xy'].shape==(117,21,2)
    assigned=[]
    for f in folds:
        train=set(f['train_indices']);val=set(f['val_indices']);test=set(f['test_indices'])
        assert not(train&val or train&test or val&test)
        groups=[{meta[i]['recording'] for i in s} for s in [train,val,test]]
        assert not(groups[0]&groups[1] or groups[0]&groups[2] or groups[1]&groups[2])
        assigned.extend(test)
    assert sorted(assigned)==list(range(117))
    assert np.all((gaze['timestamps_ns']<=gaze['cutoff_ns'][:,None])|~gaze['valid'])
    assert np.all((gaze['timestamps_ns']>=gaze['start_ns'][:,None])|~gaze['valid'])
    seed_all(17);model=Predictor().to(DEVICE).eval();idx=np.arange(117)
    truncated={k:v[:,:6].copy() if k in ['xy','H','cam','seconds'] else v.copy() for k,v in data.items()}
    maximum=0.
    for use in [False,True]:
        a=infer(model,data,gaze,idx,use);b=infer(model,truncated,gaze,idx,use)
        assert np.array_equal(a,b);maximum=max(maximum,float(np.max(np.abs(a-b))))
    corrupt={k:v.copy() for k,v in gaze.items()}
    corrupt['values'][:]=12345.;corrupt['valid'][:]=~corrupt['valid']
    assert np.array_equal(infer(model,data,gaze,idx,False),infer(model,data,corrupt,idx,False))
    assert not np.array_equal(infer(model,data,gaze,idx,True),infer(model,data,corrupt,idx,True))
    # The missing-gaze fallback is finite, even when an entire batch has no observations.
    corrupt['valid'][:]=False
    assert np.isfinite(infer(model,data,corrupt,idx,True)).all()
    result=dict(passed=True,registered_parameters=sum(p.numel() for p in model.parameters()),
        future_truncation_max_px=maximum,without_gaze_input_invariance=True,folds_recording_disjoint=True,
        protected_files=len(read(OUT/'protected_hashes.json')),code_hashes=code_hashes())
    write(OUT/'preflight.json',result);print('PREFLIGHT',result['registered_parameters'],flush=True)

def fit(data,meta,gaze,fold,seed,arm,cfg):
    dest=OUT/f'fold_{fold["fold"]}/seed_{seed}/{arm}';dest.mkdir(parents=True,exist_ok=True)
    if (dest/'complete.json').exists():
        assert read(dest/'complete.json')['code_hashes']==code_hashes();return
    begin=time.monotonic();seed_all(seed);model=Predictor().to(DEVICE)
    initial_hash=hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in model.state_dict().values())).hexdigest()
    opt=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    best=float('inf');logs=[];use=arm=='with_gaze'; draws=hashlib.sha256()
    for epoch in range(cfg['epochs']):
        model.train();losses=[]
        for update in range(cfg['updates_per_epoch']):
            rng=np.random.default_rng(seed*100000+epoch*10+update)
            idx=balanced(fold['train_indices'],meta,cfg['batch_size'],rng)
            pred,h=forward(model,data,gaze,idx,use,'train',rng)
            draws.update(idx.tobytes());draws.update(h['seq'].tobytes())
            y=tensor(target(data,idx,h['origin'],False))
            loss=(pred-y).square().mean();assert torch.isfinite(loss)
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
            losses.append(loss.item())
        if (epoch+1)%cfg['validation_every']==0:
            model.eval();idx=np.asarray(fold['val_indices']);pred=infer(model,data,gaze,idx,use)
            val=float(np.linalg.norm(pred-data['xy'][idx,P:],axis=-1).mean())
            logs.append(dict(epoch=epoch+1,training_mse=float(np.mean(losses)),validation_ADE=val))
            if val<best:best=val;state=copy.deepcopy(model.state_dict());bestepoch=epoch+1
    model.load_state_dict(state);model.eval()
    torch.save(dict(model=state,arm=arm,fold=fold,seed=seed,epoch=bestepoch,validation_ADE=best,
                    protocol_sha256=sha(OUT/'protocol.json')),dest/'checkpoint.pt')
    idx=np.asarray(fold['test_indices']);pred=infer(model,data,gaze,idx,use)
    assert np.isfinite(pred).all()
    np.savez_compressed(dest/'predictions.npz',indices=idx,pred=pred)
    write(dest/'training.json',logs)
    record=dict(seconds=time.monotonic()-begin,epoch=bestepoch,code_hashes=code_hashes(),
        initial_state_hash=initial_hash,draws_and_augmentation_hash=draws.hexdigest(),protocol_sha256=sha(OUT/'protocol.json'))
    write(dest/'complete.json',record)
    print(f'{fold["fold"]}/{seed}/{arm}: {record["seconds"]:.1f}s, epoch {bestepoch}',flush=True)

def evaluate(data,meta,gaze,folds,cfg):
    rows=[];diffs=[];replaymax=0.;futuremax=0.;seconds=0.
    truncated={k:v[:,:6].copy() if k in ['xy','H','cam','seconds'] else v.copy() for k,v in data.items()}
    for fold in folds:
        for seed in cfg['seeds']:
            pair=[]
            for arm in cfg['arms']:
                dest=OUT/f'fold_{fold["fold"]}/seed_{seed}/{arm}'
                done=read(dest/'complete.json');pair.append(done);seconds+=done['seconds']
                ck=torch.load(dest/'checkpoint.pt',map_location=DEVICE,weights_only=False)
                model=Predictor().to(DEVICE);model.load_state_dict(ck['model']);model.eval()
                stored=dict(np.load(dest/'predictions.npz'));idx=stored['indices'];use=arm=='with_gaze'
                pred=infer(model,data,gaze,idx,use);short=infer(model,truncated,gaze,idx,use)
                replaymax=max(replaymax,float(np.max(np.abs(pred-stored['pred']))))
                futuremax=max(futuremax,float(np.max(np.abs(pred-short))))
                assert np.array_equal(pred,stored['pred']) and np.array_equal(pred,short)
                if not use:
                    fake={k:v.copy() for k,v in gaze.items()};fake['values'][:]=789;fake['valid'][:]=False
                    assert np.array_equal(pred,infer(model,data,fake,idx,False))
                err=np.linalg.norm(pred-data['xy'][idx,P:],axis=-1)
                # Independently implement Euclidean metrics using Python arithmetic.
                manual=np.array([[sum(float(a-b)**2 for a,b in zip(pp,yy))**.5 for pp,yy in zip(p,y)] for p,y in zip(pred,data['xy'][idx,P:])])
                assert np.allclose(err,manual,atol=1e-9,rtol=1e-12)
                for j,i in enumerate(idx):rows.append(dict(index=int(i),key=meta[i]['key'],recording=meta[i]['recording'],
                    fold=fold['fold'],seed=seed,arm=arm,ADE=float(err[j].mean()),FDE=float(err[j,-1]),
                    nonoverlap=bool(meta[i]['nonoverlap'])))
            assert pair[0]['initial_state_hash']==pair[1]['initial_state_hash']
            assert pair[0]['draws_and_augmentation_hash']==pair[1]['draws_and_augmentation_hash']
    with (OUT/'metrics.csv').open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summaries={}
    for arm in cfg['arms']:
        r=[x for x in rows if x['arm']==arm]
        summaries[arm]={metric:float(np.mean([x[metric] for x in r])) for metric in ['ADE','FDE']}
        summaries[arm]['per_seed']={str(s):{metric:float(np.mean([x[metric] for x in r if x['seed']==s])) for metric in ['ADE','FDE']} for s in cfg['seeds']}
        summaries[arm]['recording_equal']={metric:float(np.mean([np.mean([x[metric] for x in r if x['recording']==rec]) for rec in sorted({x['recording'] for x in r})])) for metric in ['ADE','FDE']}
        summaries[arm]['sparse']={metric:float(np.mean([x[metric] for x in r if x['nonoverlap']])) for metric in ['ADE','FDE']}
    for i in range(len(meta)):
        v=[np.mean([r['ADE'] for r in rows if r['index']==i and r['arm']==arm]) for arm in cfg['arms']]
        diffs.append(dict(index=i,key=meta[i]['key'],recording=meta[i]['recording'],delta_ADE=float(v[1]-v[0])))
    recdiff=np.array([np.mean([x['delta_ADE'] for x in diffs if x['recording']==rec]) for rec in sorted({x['recording'] for x in diffs})])
    rng=np.random.default_rng(20261008);boot=np.mean(rng.choice(recdiff,(5000,len(recdiff))),axis=1)
    hold=np.linalg.norm(data['xy'][:,P:]-data['xy'][:,P-1:P],axis=-1)
    result=dict(arms=summaries,hold_last=dict(ADE=float(hold.mean()),FDE=float(hold[:,-1].mean())),
        windows_improved_seed_mean=sum(d['delta_ADE']<0 for d in diffs),windows=len(meta),
        recording_equal_delta_bootstrap95=np.quantile(boot,[.025,.975]).tolist(),recording_deltas=recdiff.tolist(),
        uncertainty='Six-recording exploratory bootstrap; overlapping windows and seeds are not independent participants.',
        training_wall_seconds_sum=seconds)
    write(OUT/'comparison.json',result);write(OUT/'paired_windows.json',diffs)
    hashes=read(OUT/'protected_hashes.json');assert all(sha(ROOT/p)==v for p,v in hashes.items())
    assert read(OUT/'preflight.json')['code_hashes']==code_hashes()
    write(OUT/'verification.json',dict(passed=True,model_replays=36,paired_initialization_and_batches=True,
        future_physically_truncated_replays=36,replay_max_px=replaymax,future_max_px=futuremax,
        no_gaze_input_invariance_replays=18,independent_metrics=True,protected_files_unchanged=len(hashes)))
    print(json.dumps(result,indent=2),flush=True)

def main():
    data=dict(np.load(OLD/'dataset.npz'));meta=read(OLD/'samples.json');gaze=dict(np.load(OUT/'gaze.npz'))
    cfg=read(OUT/'protocol.json');folds=read(OUT/'folds.json')
    if not (OUT/'preflight.json').exists():preflight(data,meta,gaze,folds)
    else:assert read(OUT/'preflight.json')['code_hashes']==code_hashes()
    if '--preflight' in sys.argv:return
    print('TRAIN',DEVICE,torch.__version__,flush=True)
    for fold in folds:
        for seed in cfg['seeds']:
            for arm in cfg['arms']:fit(data,meta,gaze,fold,seed,arm,cfg)
    evaluate(data,meta,gaze,folds,cfg)

if __name__=='__main__':main()
