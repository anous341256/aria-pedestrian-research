"""Original factorized model plus matched self/cross adapters, with branch diagnostics."""
from pathlib import Path
import sys,importlib.util,time,copy,hashlib,csv,json
import numpy as np
from cross_model import DualAttention,torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/factorized_forecast_v1_20261002'))
from shared import input_history,targets,read,write,sha,warp,project_future,SCALE,P,F,rotmat
from model import CameraPredictor
from train import seed_all,balanced,noise_for,tensor,DEVICE
spec=importlib.util.spec_from_file_location('original_person',ROOT/'scripts/factorized_forecast_v1_20261002/network.py')
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
OLD=ROOT/'outputs/biflow_aria_v1_20261002'
BASE=ROOT/'outputs/factorized_forecast_v1_20261002'
CAM=ROOT/'outputs/camera_full_data_20261002'
OUT=ROOT/'outputs/dual_cross_attention_20261008'
ARMS=['self_control','cross_attention']

def hashes():return {p.name:sha(p) for p in Path(__file__).parent.glob('*.py')}
def group_ids(meta,idx):
    keys=[(meta[int(i)]['recording'],meta[int(i)]['timestamp_ns']) for i in idx]
    lookup={k:j for j,k in enumerate(sorted(set(keys)))}
    return torch.tensor([lookup[k] for k in keys],dtype=torch.long,device=DEVICE)
def batch(indices,meta,rng):
    # Expand each sampled anchor to its complete eligible same-time person group.
    anchors=balanced(indices,meta,48,rng);result=[]
    for i in anchors:
        result.extend(j for j in indices if meta[j]['recording']==meta[i]['recording'] and meta[j]['timestamp_ns']==meta[i]['timestamp_ns'])
    return np.asarray(result,int)
def load(fold,seed,cross):
    c=CameraPredictor();p=mod.PedestrianPredictor(False)
    c.load_state_dict(torch.load(CAM/f'fold_{fold}/seed_{seed}/full/camera.pt',map_location='cpu',weights_only=False)['model'])
    p.load_state_dict(torch.load(BASE/f'fold_{fold}/seed_{seed}/separated_dual/checkpoint.pt',map_location='cpu',weights_only=False)['model'])
    seed_all(seed)
    return DualAttention(c,p,cross).to(DEVICE)
def args(data,meta,idx,mode='reviewed',rng=None):
    h=input_history(data,idx,mode,rng)
    return dict(seq=tensor(h['seq']),origin=tensor(h['origin']),cam=tensor(data['cam'][idx,:P]),group=group_ids(meta,idx)),h
@torch.no_grad()
def predict(model,data,meta,idx):
    a,h=args(data,meta,idx);p,r=model.predict(**a,noise=tensor(noise_for(meta,idx,'future',5)))
    component=(p.cpu().numpy()+h['origin'][:,None])*SCALE
    return dict(component=component,rotation=r.cpu().numpy())
def angle(pred,true):
    import cv2
    a=rotmat(pred.astype(np.float64).reshape(-1,3));b=rotmat(true.astype(np.float64).reshape(-1,3))
    values=[np.linalg.norm(cv2.Rodrigues(x.T@y)[0]) for x,y in zip(a,b)]
    return np.rad2deg(values).reshape(pred.shape[:2])
def quaternion_matrix(rv):
    # Independent quaternion expansion, separate from production cv2 Rodrigues.
    rv=np.asarray(rv,dtype=np.float64);theta=np.linalg.norm(rv,axis=-1)
    xyz=.5*np.sinc(theta[...,None]/(2*np.pi))*rv;w=np.cos(theta/2)
    x,y,z=xyz.T
    return np.stack([1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w),
        2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w),
        2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)],axis=-1).reshape(-1,3,3)
def project(component,rotation,K):
    pairs=[project_future(p,r,k) for p,r,k in zip(component,rotation,K)]
    return np.stack([x[0] for x in pairs]),np.stack([x[1] for x in pairs])
def diagnostics(z,data,idx):
    truth=warp(data['xy'][idx,P:],data['H'][idx,P:])[0]
    z=dict(z)
    z['image'],z['depth']=project(z['component'],z['rotation'],data['K'][idx])
    z['gt_camera'],z['gt_camera_depth']=project(z['component'],data['cam'][idx,P:],data['K'][idx])
    z['gt_person'],z['gt_person_depth']=project(truth,z['rotation'],data['K'][idx])
    z['angle_deg']=angle(z['rotation'],data['cam'][idx,P:])
    z['residual_error']=np.linalg.norm(z['component']-truth,axis=-1)
    for name in ['image','gt_camera','gt_person']:
        depthname='depth' if name=='image' else name+'_depth'
        err=np.linalg.norm(z[name]-data['xy'][idx,P:],axis=-1)
        bad=(z[depthname]<=1e-6)|~np.isfinite(err);err[bad]=32000.
        z[name+'_error']=err
    return z
def get_scales(model,data,meta,idx):
    z=diagnostics(predict(model,data,meta,idx),data,idx)
    truth=warp(data['xy'][idx,P:],data['H'][idx,P:])[0]
    group=group_ids(meta,idx).cpu().numpy()
    unique=[np.flatnonzero(group==g)[0] for g in np.unique(group)]
    return dict(person_mse=max(1e-5,float(np.mean(((z['component']-truth)/SCALE)**2))),
        camera_mse=max(1e-4,float(np.mean((z['rotation'][unique]-data['cam'][idx,P:][unique])**2))),
        residual_ADE=max(10.,float(z['residual_error'].mean())),
        camera_angle=max(1.,float(z['angle_deg'][unique].mean())))
def score(z,meta,idx,scales):
    group=group_ids(meta,idx).cpu().numpy();unique=[np.flatnonzero(group==g)[0] for g in np.unique(group)]
    return float(z['residual_error'].mean()/scales['residual_ADE']+z['angle_deg'][unique].mean()/scales['camera_angle'])

def prepare(data,meta):
    OUT.mkdir(parents=True,exist_ok=True)
    cfg=dict(version='dual_cross_attention_20261008',arms=ARMS,seeds=[17,29,43],windows=117,folds=6,
        updates=240,validation_every=20,anchor_batch=48,lr=.001,weight_decay=.01,clip=1.,
        history_weight=.5,noise_sigma=.15,inference_samples=5,flow_steps=10,
        adaptation='Freeze original separated GRU encoders, flow future/history heads, context, full-data GRU camera head. Train only zero-initialized attention residual adapters; initial output equals old model.',
        control='Identical adapter weights/count; self_control reads own stream, cross_attention reads other stream. Same paired batches/augmentation/noise/optimizer.',
        group_rule='Camera cross-attends to mean person GRU sequence for all eligible dataset targets at identical recording/timestamp; output shared within group. Expand all batch anchors to complete groups. Camera loss/validation each time group one vote.',
        supervision='Flow future MSE +0.5 history MSE, divided by training-base normalized residual MSE; camera rotation-vector MSE divided by training-base MSE. No composed-image loss.',
        selection='Minimum sum of validation residual ADE/training-base ADE and camera mean angle/training-base angle. Epoch0 eligible in both arms. Same joint selected checkpoint for both branches.',
        comparison='Frozen original is reference; self-vs-cross isolates attention context. Versus original also changes capacity, adaptation exposure and selection.',
        interpretation='Residual contains camera translation/parallax. Same-time target census is existing offline cohort, not an online full-scene detector. No gaze.',no_test_tuning=True)
    if (OUT/'protocol.json').exists():assert read(OUT/'protocol.json')==cfg
    else:write(OUT/'protocol.json',cfg)
    folds=read(BASE/'folds.json');write(OUT/'folds.json',folds)
    protected=[OLD/'dataset.npz',OLD/'samples.json',BASE/'folds.json']
    for f in folds:
        split=[set(f[k]) for k in ['train_indices','val_indices','test_indices']]
        groups=[{meta[i]['recording'] for i in ids} for ids in split]
        assert not(groups[0]&groups[1] or groups[0]&groups[2] or groups[1]&groups[2])
        for s in cfg['seeds']:
            protected += [CAM/f'fold_{f["fold"]}/seed_{s}/full/camera.pt',BASE/f'fold_{f["fold"]}/seed_{s}/separated_dual/checkpoint.pt',BASE/f'fold_{f["fold"]}/seed_{s}/separated_dual/predictions_reviewed.npz']
    protected += list((ROOT/'scripts/factorized_forecast_v1_20261002').glob('*.py'))
    if not (OUT/'source_hashes.json').exists():write(OUT/'source_hashes.json',{str(p.relative_to(ROOT)):sha(p) for p in protected})
    # A shared camera observation/GT must actually be shared in the source arrays.
    allgroups=group_ids(meta,np.arange(len(meta))).cpu().numpy()
    for g in np.unique(allgroups):
        ii=np.flatnonzero(allgroups==g)
        assert np.max(np.abs(data['cam'][ii]-data['cam'][ii[0]]))<1e-6
    write(OUT/'preflight.json',dict(passed=True,code_hashes=hashes(),unique_recording_times=int(len(np.unique(allgroups)))))
    return cfg,folds

def fit(data,meta,fold,seed,arm,cfg):
    dest=OUT/f'fold_{fold["fold"]}/seed_{seed}/{arm}';dest.mkdir(parents=True,exist_ok=True)
    if (dest/'complete.json').exists():assert read(dest/'complete.json')['code_hashes']==hashes();return
    begin=time.monotonic();model=load(fold['fold'],seed,arm=='cross_attention');model.eval()
    initial=hashlib.sha256(b''.join(v.cpu().numpy().tobytes() for v in model.state_dict().values())).hexdigest()
    scales=get_scales(model,data,meta,np.asarray(fold['train_indices']))
    idx=np.asarray(fold['val_indices']);z=diagnostics(predict(model,data,meta,idx),data,idx)
    best=score(z,meta,idx,scales);state=copy.deepcopy(model.state_dict());beststep=0
    logs=[dict(step=0,validation_score=best,residual_ADE=float(z['residual_error'].mean()),camera_angle=float(z['angle_deg'].mean()))]
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=cfg['lr'],weight_decay=cfg['weight_decay'])
    draws=hashlib.sha256();losses=[];batch_sizes=[]
    for step in range(1,cfg['updates']+1):
        model.train();rng=np.random.default_rng(seed*100000+step)
        idx=batch(fold['train_indices'],meta,rng);a,h=args(data,meta,idx,'train',rng)
        y=targets(data,idx,h['origin']);n=len(idx);batch_sizes.append(n)
        t1=rng.random(n);t2=rng.random(n);n1=rng.normal(0,.15,(n,P*2));n2=rng.normal(0,.15,(n,F*2))
        for v in [idx,h['seq'],t1,t2,n1,n2]:draws.update(v.tobytes())
        loss,parts=model.loss(**a,past=tensor(y['past']),future=tensor(y['future']),rotation=tensor(data['cam'][idx,P:]),
            t1=tensor(t1),t2=tensor(t2),n1=tensor(n1),n2=tensor(n2),scales=scales)
        assert torch.isfinite(loss);opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
        losses.append([loss.item()]+[p.item() for p in parts])
        if step%cfg['validation_every']==0:
            model.eval();idx=np.asarray(fold['val_indices']);z=diagnostics(predict(model,data,meta,idx),data,idx)
            val=score(z,meta,idx,scales)
            logs.append(dict(step=step,validation_score=val,residual_ADE=float(z['residual_error'].mean()),camera_angle=float(z['angle_deg'].mean()),losses=np.mean(losses[-20:],axis=0).tolist()))
            if val<best:best=val;beststep=step;state=copy.deepcopy(model.state_dict())
    model.load_state_dict(state);model.eval();idx=np.asarray(fold['test_indices'])
    torch.save(dict(model=state,fold=fold,seed=seed,arm=arm,step=beststep,scales=scales,protocol_hash=sha(OUT/'protocol.json')),dest/'checkpoint.pt')
    z=diagnostics(predict(model,data,meta,idx),data,idx);np.savez_compressed(dest/'predictions.npz',indices=idx,**z)
    write(dest/'training.json',logs)
    write(dest/'complete.json',dict(seconds=time.monotonic()-begin,step=beststep,initial_hash=initial,draws_hash=draws.hexdigest(),
        code_hashes=hashes(),trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),batch_range=[min(batch_sizes),max(batch_sizes)]))
    print(f'{fold["fold"]}/{seed}/{arm} step={beststep} {time.monotonic()-begin:.1f}s',flush=True)

def evaluate(data,meta,folds,cfg):
    rows=[];maxreplay=0.;maxfuture=0.;maxinitial=0.;maxshared=0.;total=0.
    truncated={k:v[:,:P].copy() if k in ['xy','cam','H','seconds'] else v.copy() for k,v in data.items()}
    for fold in folds:
        idx=np.asarray(fold['test_indices'])
        for seed in cfg['seeds']:
            base=load(fold['fold'],seed,False).eval();original=predict(base,data,meta,idx)
            old=dict(np.load(BASE/f'fold_{fold["fold"]}/seed_{seed}/separated_dual/predictions_reviewed.npz'))
            assert len(old['component'])==len(idx)
            cached_rotation=np.load(CAM/f'fold_{fold["fold"]}/seed_{seed}/full/predictions.npz')['person_rotation'][idx]
            assert np.max(np.abs(original['rotation']-cached_rotation))<1e-6
            d=float(np.max(np.abs(original['component']-old['component'])));maxinitial=max(maxinitial,d);assert d<.002
            cross0=predict(load(fold['fold'],seed,True).eval(),data,meta,idx)
            assert np.array_equal(original['component'],cross0['component']) and np.array_equal(original['rotation'],cross0['rotation'])
            records=[]
            for arm in ['original']+ARMS:
                if arm=='original':z=diagnostics(original,data,idx)
                else:
                    dest=OUT/f'fold_{fold["fold"]}/seed_{seed}/{arm}'
                    complete=read(dest/'complete.json');records.append(complete);total+=complete['seconds']
                    ck=torch.load(dest/'checkpoint.pt',map_location=DEVICE,weights_only=False)
                    model=load(fold['fold'],seed,arm=='cross_attention');model.load_state_dict(ck['model']);model.eval()
                    # Source backbone tensors are unchanged, including the frozen decoders.
                    for key,value in base.state_dict().items():
                        if key.startswith(('camera.','person.')):assert torch.equal(value,model.state_dict()[key])
                    actual=predict(model,data,meta,idx);short=predict(model,truncated,meta,idx)
                    stored=dict(np.load(dest/'predictions.npz'))
                    for key in ['component','rotation']:
                        maxreplay=max(maxreplay,float(np.max(np.abs(actual[key]-stored[key]))))
                        maxfuture=max(maxfuture,float(np.max(np.abs(actual[key]-short[key]))))
                        assert np.array_equal(actual[key],stored[key]) and np.array_equal(actual[key],short[key])
                    z=diagnostics(actual,data,idx)
                groups=group_ids(meta,idx).cpu().numpy()
                for g in np.unique(groups):
                    ids=np.flatnonzero(groups==g);dd=float(np.max(np.abs(z['rotation'][ids]-z['rotation'][ids[0]])))
                    maxshared=max(maxshared,dd);assert dd<1e-6
                # Independent quaternion/K projection checks geometric composition.
                for j,i in enumerate(idx):
                    rays=np.c_[z['component'][j],np.ones(F)]@np.linalg.inv(data['K'][i]).T
                    transformed=np.einsum('tji,tj->ti',quaternion_matrix(z['rotation'][j]),rays)@data['K'][i].T
                    xy=transformed[:,:2]/transformed[:,2:]
                    assert np.max(np.abs(xy-z['image'][j]))<.01
                    row=dict(index=int(i),key=meta[i]['key'],recording=meta[i]['recording'],timestamp_ns=meta[i]['timestamp_ns'],seed=seed,arm=arm,nonoverlap=meta[i]['nonoverlap'])
                    for name,errors in [('image',z['image_error']),('residual',z['residual_error']),('gt_camera',z['gt_camera_error']),('gt_person',z['gt_person_error']),('camera_deg',z['angle_deg'])]:
                        row[name+'_ADE']=float(errors[j].mean());row[name+'_FDE']=float(errors[j,-1])
                    row['invalid']=int((z['depth'][j]<=1e-6).sum());rows.append(row)
            assert records[0]['initial_hash']==records[1]['initial_hash'] and records[0]['draws_hash']==records[1]['draws_hash']
    fields=list(rows[0]);metrics=[k for k in fields if k.endswith(('_ADE','_FDE'))]
    with (OUT/'metrics.csv').open('w',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    summary={}
    for arm in ['original']+ARMS:
        rr=[r for r in rows if r['arm']==arm]
        summary[arm]={k:float(np.mean([r[k] for r in rr])) for k in metrics}
        summary[arm]['per_seed']={str(s):{k:float(np.mean([r[k] for r in rr if r['seed']==s])) for k in metrics} for s in cfg['seeds']}
        summary[arm]['recording_equal']={k:float(np.mean([np.mean([r[k] for r in rr if r['recording']==rec]) for rec in sorted({r['recording'] for r in rr})])) for k in metrics}
        summary[arm]['sparse']={k:float(np.mean([r[k] for r in rr if r['nonoverlap']])) for k in metrics}
        # Camera error also reported once per source-time/seed, not once per person.
        unique={ (r['recording'],r['timestamp_ns'],r['seed']):r for r in rr }
        summary[arm]['camera_time_equal']={k:float(np.mean([r[k] for r in unique.values()])) for k in ['camera_deg_ADE','camera_deg_FDE']}
        summary[arm]['invalid']=sum(r['invalid'] for r in rr)
    hold=np.linalg.norm(data['xy'][:,P:]-data['xy'][:,P-1:P],axis=-1)
    write(OUT/'comparison.json',dict(arms=summary,hold=dict(ADE=float(hold.mean()),FDE=float(hold[:,-1].mean())),training_seconds=total))
    protected=read(OUT/'source_hashes.json');assert all(sha(ROOT/k)==v for k,v in protected.items())
    assert read(OUT/'preflight.json')['code_hashes']==hashes()
    write(OUT/'verification.json',dict(passed=True,models=36,original_initialization_max_px=maxinitial,replay_max_px=maxreplay,
        future_truncation_max=maxfuture,shared_camera_max_rad=maxshared,independent_quaternion_projection=True,
        frozen_backbones_unchanged=True,paired_initialization_and_draws=True,protected_files_unchanged=len(protected)))
    print(json.dumps({k:{n:v for n,v in val.items() if isinstance(v,(float,int))} for k,val in summary.items()},indent=2),flush=True)

def main():
    data=dict(np.load(OLD/'dataset.npz'));meta=read(OLD/'samples.json');cfg,folds=prepare(data,meta)
    if '--preflight' in sys.argv:
        f=folds[0];idx=np.asarray(f['test_indices'])
        z=predict(load(0,17,False).eval(),data,meta,idx)
        zz=predict(load(0,17,True).eval(),data,meta,idx)
        old=np.load(BASE/'fold_0/seed_17/separated_dual/predictions_reviewed.npz')
        assert np.max(np.abs(z['component']-old['component']))<.002
        assert np.array_equal(z['component'],zz['component']) and np.array_equal(z['rotation'],zz['rotation'])
        print('PREFLIGHT PASSED: zero adapters reproduce old model',flush=True);return
    print('TRAIN',DEVICE,flush=True)
    for fold in folds:
        for seed in cfg['seeds']:
            for arm in ARMS:fit(data,meta,fold,seed,arm,cfg)
    evaluate(data,meta,folds,cfg)
if __name__=='__main__':main()
