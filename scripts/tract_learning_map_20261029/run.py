"""Replay existing optimization exactly; add train-only TrACT-style diagnostics."""
from pathlib import Path
import sys, importlib.util, json, hashlib, copy, time, random
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
VERSION='tract_learning_map_20261029'
OUT=ROOT/'outputs'/VERSION
SRC=ROOT/'outputs/horizon_15_15_g1_20261029'
spec=importlib.util.spec_from_file_location('original_long_training',ROOT/'scripts/horizon_15_15_g1_20261029/train_long.py')
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
b=old.base;torch=old.torch;read=b.read;write=b.write;sha=b.sha
STEPS=np.arange(0,241,20)

def protected():
    paths=list(SRC.rglob('checkpoint.pt'))+list(SRC.rglob('complete.json'))+list(SRC.rglob('training.json'))
    paths += [SRC/x for x in ['dataset.npz','samples.json','folds.json','protocol.json','comparison.json','metrics.json']]
    paths += list((ROOT/'scripts/horizon_15_15_g1_20261029').glob('*.py'))
    paths += list((ROOT/'scripts/translation_forecast_20261020').glob('*.py'))
    return {str(p.relative_to(ROOT)):sha(p) for p in paths}

def edgc(d,meta):
    rows=[]
    for i,m in enumerate(meta):
        t=d['seconds'][i,:16].astype(float);tt=t-t.mean();den=tt@tt
        height=d['boxes'][i,:16,3]-d['boxes'][i,:16,1]
        xy=b.warp(d['xy'][i,:16],d['H'][i,:16])[0]
        p=d['translation'][i,:16].astype(float)
        values=dict(E=float(tt@np.log(height)/den),D=float(np.median(height)/1920),
                    G=float(tt@xy[:,1]/den/1920),C=float(np.linalg.norm(np.diff(p,axis=0),axis=1).sum()/(t[-1]-t[0])))
        assert all(np.isfinite(list(values.values()))) and np.all(height>0)
        rows.append(dict(index=i,key=m['key'],recording=m['recording'],source='old' if i<65 else 'new',features=values))
    return rows

def rng_snapshot():
    return (random.getstate(),np.random.get_state(),torch.get_rng_state().clone(),[s.clone() for s in torch.cuda.get_rng_state_all()])

def same_rng(a,c):
    return a[0]==c[0] and all(np.array_equal(x,y) for x,y in zip(a[1],c[1])) and torch.equal(a[2],c[2]) and all(torch.equal(x,y) for x,y in zip(a[3],c[3]))

@torch.no_grad()
def snapshot(pm,cm,d,meta,idx):
    before=rng_snapshot();pm.eval();z=old.predict(pm,cm,d,meta,idx)
    truth=b.warp(d['xy'][idx,16:31],d['H'][idx,16:31])[0]
    image,depth=b.project(z['component'],z['rotation'],d['K'][idx])
    e=np.linalg.norm(image-d['xy'][idx,16:31],axis=-1)
    invalid=(depth<=1e-6)|~np.isfinite(e);e[invalid]=32000
    result=dict(residual=np.linalg.norm(z['component']-truth,axis=-1),image=e,invalid=invalid)
    assert same_rng(before,rng_snapshot()),'Diagnostic inference changed RNG'
    return result

def fit(d,meta,f,seed,arm,cm):
    dest=OUT/arm/f'fold_{f["fold"]}'/f'seed_{seed}'
    if (dest/'complete.json').exists():
        assert read(dest/'complete.json')['runner_sha']==sha(Path(__file__))
        return read(dest/'complete.json')
    dest.mkdir(parents=True,exist_ok=True)
    b.seed_all(seed);m=old.Person().to(b.DEVICE);initial=b.statehash(m)
    train=f['train_indices'] if arm=='original_long' else f['expanded_train_indices']
    monitor=np.array(f['expanded_train_indices']);vi=np.array(f['val_indices'])
    assert not {meta[i]['recording'] for i in monitor}&{f['test'],f['validation']}
    stable=b.warp(d['xy'],d['H'])[0]/b.SCALE
    opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.01)
    best=float('inf');state=None;logs=[];counts=np.zeros(len(meta),int);exposure=[]
    curves={key:[] for key in ['residual','image','invalid']};valcurves={key:[] for key in curves}
    losslog=[];drawhash=hashlib.sha256();start=time.monotonic()
    for step in range(241):
        if step:
            epoch,update=divmod(step-1,2);rng=np.random.default_rng(seed*100000+epoch*10+update)
            idx=b.balanced(train,meta,rng);drawhash.update(idx.tobytes());np.add.at(counts,idx,1)
            seq,o=b.features(d,idx,'with_translation');y=stable[idx]-o[:,None]
            t1=b.tensor(rng.random(48));t2=b.tensor(rng.random(48))
            n1=b.tensor(rng.normal(0,.15,(48,32)));n2=b.tensor(rng.normal(0,.15,(48,30)))
            m.train();loss,lf,lh=m.loss(b.tensor(seq),b.tensor(o),b.tensor(y[:,:16].reshape(48,-1)),b.tensor(y[:,16:31].reshape(48,-1)),t1,t2,n1,n2)
            assert torch.isfinite(loss);opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step()
            losslog.append([step,float(loss.detach()),float(lf.detach()),float(lh.detach())])
        if step%20==0:
            # Match original validation arithmetic/selection exactly.
            m.eval();z=old.predict(m,cm,d,meta,vi)
            value=float(np.linalg.norm(z['component']-stable[vi,16:31]*b.SCALE,axis=-1).mean())
            logs.append(dict(step=step,validation_residual_ADE=value))
            if value<best:best=value;beststep=step;state=copy.deepcopy(m.state_dict())
            for buffer,indices in [(curves,monitor),(valcurves,vi)]:
                zz=snapshot(m,cm,d,meta,indices)
                for key in buffer:buffer[key].append(zz[key])
            exposure.append(counts.copy())
            torch.save(dict(model=m.state_dict(),step=step),dest/f'step_{step:03d}.pt')
    np.savez_compressed(dest/'curves.npz',steps=STEPS,indices=monitor,val_indices=vi,train_indices=np.array(train),
        exposure=np.array(exposure),losses=np.array(losslog),**{k:np.array(v) for k,v in curves.items()},
        **{'val_'+k:np.array(v) for k,v in valcurves.items()})
    write(dest/'training.json',logs)
    source=SRC/arm/f'fold_{f["fold"]}'/f'seed_{seed}'
    prior=read(source/'complete.json');priorcp=torch.load(source/'checkpoint.pt',map_location=b.DEVICE,weights_only=False)
    assert beststep==prior['step'] and initial==prior['initial_hash'] and drawhash.hexdigest()==prior['draws_hash']
    assert logs==read(source/'training.json'),'Original validation curve not reproduced'
    assert all(torch.equal(v,priorcp['model'][k]) for k,v in state.items()),'Selected parameters changed'
    m.load_state_dict(state);m.eval();idx=np.array(f['test_indices'])
    replay=old.diagnostics(old.predict(m,cm,d,meta,idx),d,idx);saved=dict(np.load(source/'predictions.npz'))
    assert all(np.array_equal(v,saved[k]) for k,v in replay.items())
    info=dict(fold=f['fold'],seed=seed,arm=arm,step=beststep,seconds=time.monotonic()-start,
              runner_sha=sha(Path(__file__)),initial_hash=initial,draws_hash=drawhash.hexdigest(),
              exact_original_weights=True,exact_original_validation=True,exact_original_test_replay=True,
              actual_old_draws=int(counts[:65].sum()),actual_new_draws=int(counts[65:].sum()),
              invalid_monitor_rays=int(np.sum(curves['invalid'])),invalid_validation_rays=int(np.sum(valcurves['invalid'])))
    write(dest/'complete.json',info);print('FIT',arm,f['fold'],seed,'selected',beststep,'exact',flush=True)
    return info

def main():
    OUT.mkdir(parents=True,exist_ok=True);cfg=read(SRC/'protocol.json')
    pre=OUT/'source_hashes.json'
    if not pre.exists():write(pre,protected())
    write(OUT/'protocol.json',dict(source=str(SRC),plan=(ROOT/'plans/TRACT_LEARNING_MAP_20261029.txt').read_text(),
                                  runner_sha=sha(Path(__file__)),checkpoints=STEPS.tolist(),seeds=cfg['seeds']))
    d=dict(np.load(SRC/'dataset.npz'));meta=read(SRC/'samples.json');folds=read(SRC/'folds.json')
    write(OUT/'edgc_features.json',edgc(d,meta))
    checks=[]
    for f in folds:
        for seed in cfg['seeds']:
            cm=old.Camera().to(b.DEVICE);cp=SRC/'camera'/f'fold_{f["fold"]}'/f'seed_{seed}'/'checkpoint.pt'
            cm.load_state_dict(torch.load(cp,map_location=b.DEVICE,weights_only=False)['model']);cm.eval()
            for arm in cfg['arms']:
                write(OUT/'progress.json',dict(fold=f['fold'],seed=seed,arm=arm,finished=len(checks)))
                checks.append(fit(d,meta,f,seed,arm,cm))
    hashes=read(pre);assert all(sha(ROOT/p)==h for p,h in hashes.items())
    write(OUT/'verification.json',dict(passed=True,fits=checks,protected_files=len(hashes),
        train_diagnostics_exclude_validation_test=True,no_test_learning_curves=True,
        instrumentation_rng_unchanged=True,training_seconds=sum(c['seconds'] for c in checks)))
    print('COMPLETE',len(checks),flush=True)
if __name__=='__main__':main()
