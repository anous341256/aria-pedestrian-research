"""Train and audit the locked horizon swap; never alter prior runs."""
import os,sys,time,copy,hashlib,random,csv
from pathlib import Path
import numpy as np
from prepare import ROOT,OUT,OLD,BASE,CAM,read,write,sha,rm,rv,warp
from network import Camera,Person,torch
torch.set_num_threads(2)
DEVICE='cuda' if torch.cuda.is_available() else 'cpu'
SCALE=np.array([2560.,1920.]);ARMS=['long_history','short_history_control']
def tensor(x):return torch.as_tensor(x,dtype=torch.float32,device=DEVICE)
def seed_all(s):random.seed(s);np.random.seed(s);torch.manual_seed(s);torch.cuda.manual_seed_all(s)
def codehash():return {n:sha(Path(__file__).with_name(n)) for n in ['prepare.py','network.py','run.py']}
def statehash(model,skip_history=False):return hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for k,v in model.state_dict().items() if not(skip_history and k.startswith('history.')))).hexdigest()
def start(arm):return 0 if arm=='long_history' else 10
def balanced(indices,meta,rng):
 groups=sorted({meta[i]['recording'] for i in indices});return np.array([rng.choice([i for i in indices if meta[i]['recording']==g]) for g in rng.choice(groups,48)])
def features(data,idx,arm,train=False,rng=None):
 a=start(arm);P=16-a;obs=data['xy'][idx,a:16].copy();valid=np.ones(obs.shape[:2])
 if train:
  choice=rng.random(len(idx));old=(choice>=.4)&(choice<.6);noisy=choice>=.6
  obs[old]=data['raw'][idx,a:16][old];valid[old]=data['raw_valid'][idx,a:16][old]
  jitter=rng.normal(0,.005,obs.shape)*SCALE;obs[noisy]+=jitter[noisy];keep=(rng.random(valid.shape)>=.2).astype(float);keep[:,-1]=1;valid[noisy]=keep[noisy]
 origin=obs[:,-1]/SCALE;stable=warp(obs,data['H'][idx,a:16])[0]/SCALE;rel=stable-origin[:,None];delta=np.zeros_like(rel);delta[:,1:]=(stable[:,1:]-stable[:,:-1])*(valid[:,1:]*valid[:,:-1])[...,None];rel[valid==0]=0
 seq=np.concatenate([rel,delta,valid[...,None],data['seconds'][idx,a:16,None]/((P-1)*.1)],-1)
 return seq.astype(np.float32),origin.astype(np.float32)
def noise(meta,idx):return np.stack([np.random.default_rng(100000+int(hashlib.sha256((meta[i]['key']+'future').encode()).hexdigest()[:8],16)).normal(0,.15,(5,10)) for i in idx]).astype(np.float32)
@torch.no_grad()
def infer_camera(model,x):return np.concatenate([model(tensor(y)).cpu().numpy() for y in np.array_split(x,max(1,int(np.ceil(len(x)/1024))))])
@torch.no_grad()
def predict(person,camera,data,meta,idx,arm):
 seq,origin=features(data,idx,arm);pc=person.predict(tensor(seq),tensor(origin),tensor(noise(meta,idx))).cpu().numpy()
 return {'component':(pc+origin[:,None])*SCALE,'rotation':infer_camera(camera,data['cam'][idx,start(arm):16])}
def project(x,r,K):
 H=np.einsum('nij,ntkj,nkl->ntil',K,rm(r),np.linalg.inv(K));return warp(x,H)
def diagnostics(z,data,idx):
 z=dict(z);target=data['xy'][idx,16:21];truth=warp(target,data['H'][idx,16:21])[0];z['residual_error']=np.linalg.norm(z['component']-truth,axis=-1)
 z['angle_deg']=np.rad2deg(np.linalg.norm(rv(np.einsum('ntji,ntjk->ntik',rm(z['rotation']),rm(data['cam'][idx,16:21]))),axis=-1))
 for name,x,r in [('image',z['component'],z['rotation']),('gt_camera',z['component'],data['cam'][idx,16:21]),('gt_person',truth,z['rotation']),('frozen_camera',z['component'],np.zeros_like(z['rotation']))]:
  p,depth=project(x,r,data['K'][idx]);e=np.linalg.norm(p-target,axis=-1);bad=(depth<=1e-6)|~np.isfinite(e);e[bad]=32000.;z[name]=p;z[name+'_depth']=depth;z[name+'_error']=e
 return z
def camera_fit(cdata,fold,seed,arm,cfg):
 dest=OUT/arm/f'fold_{fold["fold"]}/seed_{seed}/camera';dest.mkdir(parents=True,exist_ok=True);seed_all(seed);model=Camera().to(DEVICE)
 if (dest/'complete.json').exists():
  assert read(dest/'complete.json')['code_hashes']==codehash();model.load_state_dict(torch.load(dest/'checkpoint.pt',map_location=DEVICE,weights_only=False)['model']);return model.eval()
 begin=time.monotonic();init=statehash(model);tr=tensor(cdata[fold['train_indices']]);val=cdata[fold['val_indices']];opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01);rng=np.random.default_rng(seed);left=np.array([],int);logs=[];best=float('inf');drawhash=hashlib.sha256()
 for step in range(1,2001):
  while len(left)<256:left=np.concatenate([left,rng.permutation(len(tr))])
  ii,left=left[:256],left[256:];drawhash.update(ii.tobytes());b=tr[torch.as_tensor(ii,device=DEVICE)];model.train();loss=(model(b[:,start(arm):16])-b[:,16:21]).square().mean();assert torch.isfinite(loss)
  opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
  if step%100==0:
   model.eval();v=float(np.mean((infer_camera(model,val[:,start(arm):16])-val[:,16:21])**2));logs.append(dict(step=step,validation_mse=v))
   if v<best:best=v;beststep=step;state=copy.deepcopy(model.state_dict())
   write(OUT/'progress.json',dict(arm=arm,fold=fold['fold'],seed=seed,stage='camera',step=step,total=2000,seconds=time.monotonic()-begin))
 model.load_state_dict(state);torch.save(dict(model=state,step=beststep,seed=seed,arm=arm,fold=fold,protocol_sha256=sha(OUT/'protocol.json')),dest/'checkpoint.pt');write(dest/'training.json',logs)
 write(dest/'complete.json',dict(seconds=time.monotonic()-begin,step=beststep,initial_hash=init,draws_hash=drawhash.hexdigest(),parameters=sum(p.numel() for p in model.parameters()),code_hashes=codehash()))
 print(f'{arm} fold{fold["fold"]} seed{seed} camera {time.monotonic()-begin:.1f}s step{beststep}',flush=True);return model.eval()
def person_fit(data,meta,fold,seed,arm,camera,cfg):
 dest=OUT/arm/f'fold_{fold["fold"]}/seed_{seed}/person';dest.mkdir(parents=True,exist_ok=True);P=16-start(arm);seed_all(seed);model=Person(P).to(DEVICE)
 if (dest/'complete.json').exists():
  assert read(dest/'complete.json')['code_hashes']==codehash();return
 begin=time.monotonic();init=statehash(model,True);opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01);logs=[];best=float('inf');drawhash=hashlib.sha256();stable=warp(data['xy'],data['H'])[0]/SCALE
 for step in range(240):
  epoch,update=divmod(step,2);rng=np.random.default_rng(seed*100000+epoch*10+update);idx=balanced(fold['train_indices'],meta,rng);drawhash.update(idx.tobytes());seq,origin=features(data,idx,arm,True,rng);y=stable[idx]-origin[:,None]
  t1=tensor(rng.random(48));t2=tensor(rng.random(48));n1=tensor(rng.normal(0,.15,(48,P*2)));n2=tensor(rng.normal(0,.15,(48,10)))
  model.train();loss,lf,lh=model.loss(tensor(seq),tensor(origin),tensor(y[:,start(arm):16].reshape(48,-1)),tensor(y[:,16:21].reshape(48,-1)),t1,t2,n1,n2);assert torch.isfinite(loss)
  opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
  if (step+1)%20==0:
   model.eval();idx=np.asarray(fold['val_indices']);z=predict(model,camera,data,meta,idx,arm);v=float(np.linalg.norm(z['component']-stable[idx,16:21]*SCALE,axis=-1).mean());logs.append(dict(step=step+1,validation_residual_ADE=v))
   if v<best:best=v;beststep=step+1;state=copy.deepcopy(model.state_dict())
 model.load_state_dict(state);model.eval();torch.save(dict(model=state,step=beststep,seed=seed,arm=arm,fold=fold,protocol_sha256=sha(OUT/'protocol.json')),dest/'checkpoint.pt');write(dest/'training.json',logs)
 idx=np.asarray(fold['test_indices']);z=diagnostics(predict(model,camera,data,meta,idx,arm),data,idx);np.savez_compressed(dest/'predictions.npz',indices=idx,**z)
 write(dest/'complete.json',dict(seconds=time.monotonic()-begin,step=beststep,shared_initial_hash=init,draws_hash=drawhash.hexdigest(),parameters=sum(p.numel() for p in model.parameters()),inference_parameters=sum(p.numel() for k,p in model.named_parameters() if not k.startswith('history.')),code_hashes=codehash()))
 print(f'{arm} fold{fold["fold"]} seed{seed} person {time.monotonic()-begin:.1f}s step{beststep}',flush=True)
def summary(rows):
 keys=['ADE','FDE','residual_ADE','residual_FDE','camera_mean_deg','camera_final_deg','gt_camera_ADE','gt_person_ADE','frozen_camera_ADE']
 keys=[k for k in keys if k in rows[0]]
 def mean(rr):return {k:float(np.mean([r[k] for r in rr])) for k in keys}
 rec={g:mean([r for r in rows if r['recording']==g]) for g in sorted(set(r['recording'] for r in rows))}
 return dict(window_mean=mean(rows),recording_equal={k:float(np.mean([v[k] for v in rec.values()])) for k in keys},per_recording=rec,per_seed={str(s):mean([r for r in rows if r['seed']==s]) for s in sorted(set(r['seed'] for r in rows))},sparse=mean([r for r in rows if r['nonoverlap']]))
def evaluate(data,meta,folds,cfg):
 rows=[];maxreplay=0.;maxfuture=0.;maxshared=0.;seconds=0.;invalid=0;counts={}
 short={k:v[:,:16].copy() if k in ['xy','raw','raw_valid','R','H','cam','seconds','boxes'] else v.copy() for k,v in data.items()}
 for fold in folds:
  idx=np.asarray(fold['test_indices'])
  for seed in cfg['seeds']:
   for arm in ARMS:
    folder=OUT/arm/f'fold_{fold["fold"]}/seed_{seed}';cm=Camera().to(DEVICE);pm=Person(16-start(arm)).to(DEVICE)
    cm.load_state_dict(torch.load(folder/'camera/checkpoint.pt',map_location=DEVICE,weights_only=False)['model']);pm.load_state_dict(torch.load(folder/'person/checkpoint.pt',map_location=DEVICE,weights_only=False)['model']);cm.eval();pm.eval()
    stored=dict(np.load(folder/'person/predictions.npz'));z=predict(pm,cm,data,meta,idx,arm);trunc=predict(pm,cm,short,meta,idx,arm)
    for k in ['component','rotation']:
     dr=float(np.max(np.abs(z[k]-stored[k])));df=float(np.max(np.abs(z[k]-trunc[k])));maxreplay=max(maxreplay,dr);maxfuture=max(maxfuture,df);assert dr==0 and df==0
    groups={}
    for j,i in enumerate(idx):groups.setdefault((meta[i]['recording'],meta[i]['timestamp_ns']),[]).append(j)
    for jj in groups.values():maxshared=max(maxshared,float(np.max(np.abs(z['rotation'][jj]-z['rotation'][jj[0]]))))
    d=diagnostics(z,data,idx)
    for k in d:assert np.array_equal(d[k],stored[k]),k
    invalid+=int((d['image_depth']<=1e-6).sum())
    records=[read(folder/f'{part}/complete.json') for part in ['camera','person']];seconds+=sum(r['seconds'] for r in records);counts[arm]=[r['parameters'] for r in records]
    if arm==ARMS[1]:
     a=OUT/ARMS[0]/f'fold_{fold["fold"]}/seed_{seed}'
     for part in ['camera','person']:
      one=read(a/f'{part}/complete.json');two=read(folder/f'{part}/complete.json');assert one['draws_hash']==two['draws_hash'];k='initial_hash' if part=='camera' else 'shared_initial_hash';assert one[k]==two[k]
    for j,i in enumerate(idx):
     m=meta[i];rows.append(dict(arm=arm,index=int(i),key=m['key'],recording=m['recording'],seed=seed,nonoverlap=m['nonoverlap'],ADE=float(d['image_error'][j].mean()),FDE=float(d['image_error'][j,-1]),residual_ADE=float(d['residual_error'][j].mean()),residual_FDE=float(d['residual_error'][j,-1]),camera_mean_deg=float(d['angle_deg'][j].mean()),camera_final_deg=float(d['angle_deg'][j,-1]),gt_camera_ADE=float(d['gt_camera_error'][j].mean()),gt_person_ADE=float(d['gt_person_error'][j].mean()),frozen_camera_ADE=float(d['frozen_camera_error'][j].mean())))
   # Original saved0.5->1.5 system: descriptive reference only.
   olddata=np.load(OLD/'dataset.npz');cached=np.load(BASE/f'fold_{fold["fold"]}/seed_{seed}/separated_dual/predictions_reviewed.npz');assert np.array_equal(cached['indices'],idx)
   err=np.linalg.norm(cached['full']-olddata['xy'][idx,6:],axis=-1);err[(cached['full_depth']<=1e-6)|~np.isfinite(err)]=32000.
   for j,i in enumerate(idx):rows.append(dict(arm='saved_0.5_to_1.5',index=int(i),key=meta[i]['key'],recording=meta[i]['recording'],seed=seed,nonoverlap=meta[i]['nonoverlap'],ADE=float(err[j].mean()),FDE=float(err[j,-1])))
 for i,m in enumerate(meta):
  for arm in ['hold_last','raw_constant_velocity']:
   if arm=='hold_last':p=np.repeat(data['xy'][i,15:16],5,0)
   else:
    t=data['seconds'][i,10:16];xy=data['xy'][i,10:16];v=((t-t.mean())[:,None]*(xy-xy.mean(0))).sum(0)/((t-t.mean())**2).sum();p=xy[-1]+data['seconds'][i,16:21,None]*v
   e=np.linalg.norm(p-data['xy'][i,16:21],axis=-1);rows.append(dict(arm=arm,index=i,key=m['key'],recording=m['recording'],seed=0,nonoverlap=m['nonoverlap'],ADE=float(e.mean()),FDE=float(e[-1])))
 results={arm:summary([r for r in rows if r['arm']==arm]) for arm in sorted(set(r['arm'] for r in rows))};contrasts={}
 for ref in ['short_history_control','hold_last']:
  a=np.array([np.mean([r['ADE'] for r in rows if r['arm']=='long_history' and r['index']==i]) for i in range(len(meta))]);b=np.array([np.mean([r['ADE'] for r in rows if r['arm']==ref and r['index']==i]) for i in range(len(meta))]);diff=a-b
  rec=np.array([np.mean(diff[[i for i,m in enumerate(meta) if m['recording']==g]]) for g in sorted(set(m['recording'] for m in meta))]);boot=np.random.default_rng(20261017).choice(rec,(5000,len(rec)),replace=True).mean(1)
  contrasts[ref]=dict(long_minus_reference_ADE=float(diff.mean()),improved_windows=int((diff<0).sum()),improved_recordings=int((rec<0).sum()),recording_equal_difference=float(rec.mean()),exploratory_recording_bootstrap_95=np.quantile(boot,[.025,.975]).tolist())
 protected=read(OUT/'source_hashes.json');assert all(sha(ROOT/k)==v for k,v in protected.items());assert maxshared==0
 write(OUT/'metrics.json',rows);write(OUT/'comparison.json',dict(results=results,contrasts=contrasts,parameters_camera_person=counts,training_seconds=seconds))
 write(OUT/'verification.json',dict(passed=True,checkpoint_replays=72,prediction_replay_max=maxreplay,physically_future_truncated_max=maxfuture,same_time_camera_difference=maxshared,invalid_prediction_rays=invalid,paired_draws_and_shared_initialization=True,protected_files=len(protected),code_hashes=codehash()))
 print('COMPARISON', {k:v['window_mean'] for k,v in results.items()},flush=True)
def main():
 cfg=read(OUT/'protocol.json');data=dict(np.load(OUT/'dataset.npz'));meta=read(OUT/'samples.json');cdata=np.load(OUT/'camera_dataset.npz')['rotation'];folds=read(OUT/'folds.json');cfolds=read(OUT/'camera_folds.json')
 write(OUT/'execution_code_hashes.json',codehash());print('DEVICE',DEVICE,'camera_windows',len(cdata),flush=True)
 for fold,cfold in zip(folds,cfolds):
  for seed in cfg['seeds']:
   for arm in ARMS:
    cam=camera_fit(cdata,cfold,seed,arm,cfg);person_fit(data,meta,fold,seed,arm,cam,cfg)
 evaluate(data,meta,folds,cfg);write(OUT/'training_complete.json',dict(passed=True,camera_fits=36,person_fits=36))
if __name__=='__main__':main()
