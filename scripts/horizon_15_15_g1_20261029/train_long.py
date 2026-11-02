"""Matched original/expanded long-horizon training, unchanged separated family."""
from pathlib import Path
import sys,importlib.util,time,copy,hashlib,json
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/horizon_15_15_g1_20261029'
sys.path.insert(0,str(ROOT/'scripts/translation_forecast_20261020'))
spec=importlib.util.spec_from_file_location('translation_base',ROOT/'scripts/translation_forecast_20261020/run.py');base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
from network import Camera as OldCamera,Person as OldPerson
from model import FlowHead
torch=base.torch;nn=torch.nn;read=base.read;write=base.write;sha=base.sha;SCALE=base.SCALE;DEVICE=base.DEVICE;tensor=base.tensor
class Camera(OldCamera):
 def __init__(self):
  super().__init__();self.head[-1]=nn.Linear(64,45)
 def forward(self,x):return self.head(self.encoder(x)[1][-1]).reshape(-1,15,3)
class Person(OldPerson):
 def __init__(self):
  super().__init__();self.future=FlowHead(30,64)
 @torch.no_grad()
 def predict(self,seq,origin,noise):
  b,k=noise.shape[:2];h=self.encode(seq,origin)[:,None].expand(-1,k,-1).reshape(b*k,-1);x=noise.reshape(b*k,-1).clone()
  for step in range(10):
   t=torch.full((b*k,),step/10,device=x.device);estimate=self.future(x,t,h);x=x+(estimate-x)/(1-step/10)/10
  return x.reshape(b,k,15,2).mean(1)
def codehash():return {n:sha(Path(__file__).with_name(n)) for n in ['prepare_long.py','train_long.py']}
def noise(meta,idx):
 return np.stack([np.random.default_rng(100000+int(hashlib.sha256((meta[i]['key']+'future').encode()).hexdigest()[:8],16)).normal(0,.15,(5,30)) for i in idx]).astype(np.float32)
@torch.no_grad()
def predict(pm,cm,d,meta,idx):
 seq,o=base.features(d,idx,'with_translation');c=pm.predict(tensor(seq),tensor(o),tensor(noise(meta,idx))).cpu().numpy()
 return dict(component=(c+o[:,None])*SCALE,rotation=base.infer_camera(cm,base.camera_inputs(d,idx,'with_translation')))
def diagnostics(z,d,idx):
 z=dict(z);target=d['xy'][idx,16:31];truth=base.warp(target,d['H'][idx,16:31])[0];z['residual_error']=np.linalg.norm(z['component']-truth,axis=-1);z['angle_deg']=np.rad2deg(np.linalg.norm(base.rv(np.einsum('ntji,ntjk->ntik',base.rm(z['rotation']),base.rm(d['cam'][idx,16:31]))),axis=-1))
 for name,x,r in [('image',z['component'],z['rotation']),('gt_camera',z['component'],d['cam'][idx,16:31]),('gt_person',truth,z['rotation'])]:
  p,depth=base.project(x,r,d['K'][idx]);e=np.linalg.norm(p-target,axis=-1);e[(depth<=1e-6)|~np.isfinite(e)]=32000.;z[name]=p;z[name+'_depth']=depth;z[name+'_error']=e
 return z
def camera_fit(d,f,seed):
 dest=OUT/'camera'/f'fold_{f["fold"]}/seed_{seed}';dest.mkdir(parents=True,exist_ok=True);base.seed_all(seed);m=Camera().to(DEVICE)
 if (dest/'complete.json').exists():
  assert read(dest/'complete.json')['code_hashes']==codehash();m.load_state_dict(torch.load(dest/'checkpoint.pt',map_location=DEVICE,weights_only=False)['model']);return m.eval()
 start=time.monotonic();cf=f['camera'];idx=np.array(cf['train_indices']);vi=np.array(cf['val_indices']);x=tensor(base.camera_inputs(d,idx,'with_translation','rotation'));y=tensor(d['rotation'][idx,16:31]);vx=base.camera_inputs(d,vi,'with_translation','rotation');vy=d['rotation'][vi,16:31];opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.01);rng=np.random.default_rng(seed);left=np.array([],int);logs=[];best=float('inf');state=None
 for step in range(2001):
  if step:
   while len(left)<256:left=np.concatenate([left,rng.permutation(len(x))])
   ii,left=left[:256],left[256:];ii=torch.as_tensor(ii,device=DEVICE);m.train();loss=(m(x[ii])-y[ii]).square().mean();assert torch.isfinite(loss);opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step()
  if step%100==0:
   m.eval();v=float(np.mean((base.infer_camera(m,vx)-vy)**2));logs.append(dict(step=step,validation_mse=v))
   if v<best:best=v;beststep=step;state=copy.deepcopy(m.state_dict())
   write(OUT/'progress.json',dict(stage='camera',fold=f['fold'],seed=seed,step=step))
 m.load_state_dict(state);torch.save(dict(model=state,step=beststep,fold=f['fold'],seed=seed,protocol_sha256=sha(OUT/'protocol.json')),dest/'checkpoint.pt');write(dest/'training.json',logs);write(dest/'complete.json',dict(seconds=time.monotonic()-start,step=beststep,code_hashes=codehash(),parameters=sum(p.numel() for p in m.parameters())));print('CAMERA',f['fold'],seed,beststep,flush=True);return m.eval()
def person_fit(d,meta,f,seed,arm,cm):
 dest=OUT/arm/f'fold_{f["fold"]}/seed_{seed}';dest.mkdir(parents=True,exist_ok=True);base.seed_all(seed);m=Person().to(DEVICE)
 if (dest/'complete.json').exists():
  assert read(dest/'complete.json')['code_hashes']==codehash();return
 start=time.monotonic();init=base.statehash(m);train=f['train_indices'] if arm=='original_long' else f['expanded_train_indices'];vi=np.array(f['val_indices']);stable=base.warp(d['xy'],d['H'])[0]/SCALE;opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.01);best=float('inf');logs=[];drawhash=hashlib.sha256()
 for step in range(241):
  if step:
   epoch,update=divmod(step-1,2);rng=np.random.default_rng(seed*100000+epoch*10+update);idx=base.balanced(train,meta,rng);drawhash.update(idx.tobytes());seq,o=base.features(d,idx,'with_translation');y=stable[idx]-o[:,None];t1=tensor(rng.random(48));t2=tensor(rng.random(48));n1=tensor(rng.normal(0,.15,(48,32)));n2=tensor(rng.normal(0,.15,(48,30)));m.train();loss,lf,lh=m.loss(tensor(seq),tensor(o),tensor(y[:,:16].reshape(48,-1)),tensor(y[:,16:31].reshape(48,-1)),t1,t2,n1,n2);assert torch.isfinite(loss);opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step()
  if step%20==0:
   m.eval();z=predict(m,cm,d,meta,vi);v=float(np.linalg.norm(z['component']-stable[vi,16:31]*SCALE,axis=-1).mean());logs.append(dict(step=step,validation_residual_ADE=v))
   if v<best:best=v;beststep=step;state=copy.deepcopy(m.state_dict())
 m.load_state_dict(state);m.eval();torch.save(dict(model=state,step=beststep,seed=seed,fold=f['fold'],arm=arm,protocol_sha256=sha(OUT/'protocol.json')),dest/'checkpoint.pt');idx=np.array(f['test_indices']);z=diagnostics(predict(m,cm,d,meta,idx),d,idx);np.savez_compressed(dest/'predictions.npz',indices=idx,**z);write(dest/'training.json',logs);write(dest/'complete.json',dict(seconds=time.monotonic()-start,step=beststep,initial_hash=init,draws_hash=drawhash.hexdigest(),code_hashes=codehash(),parameters=sum(p.numel() for p in m.parameters())));print('PERSON',arm,f['fold'],seed,beststep,flush=True)
def row(m,i,seed,arm,z,j):
 r=base.row(m,i,seed,arm,z,j);e=z['image_error'][j];r.update(ADE_05=float(e[:5].mean()),FDE_05=float(e[4]),ADE_10=float(e[:10].mean()),FDE_10=float(e[9]));return r
def summary(rows):
 keys=[k for k in ['ADE','FDE','ADE_05','FDE_05','ADE_10','FDE_10','residual_ADE','residual_FDE','camera_mean_deg','camera_final_deg','gt_camera_ADE','gt_person_ADE'] if k in rows[0]]
 def avg(rr):return {k:float(np.mean([r[k] for r in rr])) for k in keys}
 rec={g:avg([r for r in rows if r['recording']==g]) for g in sorted({r['recording'] for r in rows})}
 return dict(window_mean=avg(rows),recording_equal={k:float(np.mean([v[k] for v in rec.values()])) for k in keys},per_recording=rec,per_seed={str(s):avg([r for r in rows if r['seed']==s]) for s in sorted({r['seed'] for r in rows})},sparse=avg([r for r in rows if r['nonoverlap']]))
def evaluate(d,meta,folds,cfg):
 rows=[];replays=0;invalid=0;shared=0.;total=0.;steps={a:0 for a in cfg['arms']};short={k:v[:,:16].copy() if k in ['xy','boxes','R','H','cam','seconds'] else v.copy() for k,v in d.items()}
 for f in folds:
  idx=np.array(f['test_indices'])
  for seed in cfg['seeds']:
   cm=Camera().to(DEVICE);cp=OUT/'camera'/f'fold_{f["fold"]}/seed_{seed}';cm.load_state_dict(torch.load(cp/'checkpoint.pt',map_location=DEVICE,weights_only=False)['model']);cm.eval();total+=read(cp/'complete.json')['seconds'];states=[]
   for arm in cfg['arms']:
    p=OUT/arm/f'fold_{f["fold"]}/seed_{seed}';pm=Person().to(DEVICE);pm.load_state_dict(torch.load(p/'checkpoint.pt',map_location=DEVICE,weights_only=False)['model']);pm.eval();z=predict(pm,cm,d,meta,idx);tr=predict(pm,cm,short,meta,idx);assert all(np.array_equal(z[k],tr[k]) for k in z);zz=diagnostics(z,d,idx);stored=dict(np.load(p/'predictions.npz'));assert all(np.array_equal(zz[k],stored[k]) for k in zz);info=read(p/'complete.json');states.append(info['initial_hash']);total+=info['seconds'];steps[arm]+=info['step']==0;invalid+=int((zz['image_depth']<=1e-6).sum());replays+=1;rows += [row(meta[i],i,seed,arm,zz,j) for j,i in enumerate(idx)]
    groups=collections_defaultdict_list()
    for j,i in enumerate(idx):groups.setdefault((meta[i]['recording'],meta[i]['timestamp_ns']),[]).append(j)
    for jj in groups.values():shared=max(shared,float(np.abs(zz['rotation'][jj]-zz['rotation'][jj[0]]).max()))
   assert states[0]==states[1]
 for i,m in enumerate(meta[:cfg['original_windows']]):
  target=d['xy'][i,16:31];hist=d['xy'][i,:16];ts=d['seconds'][i];hold=np.repeat(hist[-1,None],15,0);t=ts[:16];v=((t-t.mean())[:,None]*(hist-hist.mean(0))).sum(0)/((t-t.mean())**2).sum();cv=hist[-1]+ts[16:,None]*v
  for a,p in [('hold_last',hold),('raw_cv',cv)]:rows.append(row(m,i,0,a,{'image_error':np.linalg.norm(p-target,axis=-1)[None]},0))
 result={a:summary([r for r in rows if r['arm']==a]) for a in cfg['arms']+['hold_last','raw_cv']};contrasts={}
 for ref in ['original_long','hold_last']:
  aa=np.array([np.mean([r['ADE'] for r in rows if r['arm']=='expanded_long' and r['index']==i]) for i in range(cfg['original_windows'])]);bb=np.array([np.mean([r['ADE'] for r in rows if r['arm']==ref and r['index']==i]) for i in range(cfg['original_windows'])]);diff=aa-bb;rec=np.array([diff[[i for i,m in enumerate(meta[:cfg['original_windows']]) if m['recording']==g]].mean() for g in cfg['test_recordings']]);boot=np.random.default_rng(20261029).choice(rec,(5000,len(rec)),replace=True).mean(1);contrasts[ref]=dict(ADE_difference=float(diff.mean()),relative_percent=float((aa.mean()/bb.mean()-1)*100),improved_windows=int((diff<0).sum()),improved_recordings=int((rec<0).sum()),recording_bootstrap_95=np.quantile(boot,[.025,.975]).tolist())
 assert shared<1e-6
 protected=read(OUT/'source_hashes.json');assert all(sha(ROOT/p)==s for p,s in protected.items())
 write(OUT/'metrics.json',rows);write(OUT/'comparison.json',dict(results=result,contrasts=contrasts,training_seconds=total,step0=steps));write(OUT/'verification.json',dict(passed=True,checkpoint_replays=replays,future_physically_truncated_exact=True,paired_initializations=True,same_time_camera_max=shared,invalid_rays=invalid,protected_files=len(protected)));write(OUT/'training_complete.json',dict(passed=True,camera_fits=len(folds)*3,person_fits=replays));print('RESULT',{a:r['window_mean'] for a,r in result.items()},flush=True)
def collections_defaultdict_list():return {}
def main():
 cfg=read(OUT/'protocol.json');assert read(OUT/'preflight.json')['passed'];d=dict(np.load(OUT/'dataset.npz'));meta=read(OUT/'samples.json');cd=dict(np.load(OUT/'camera_dataset.npz'));folds=read(OUT/'folds.json');write(OUT/'execution_code_hashes.json',codehash());print('DEVICE',DEVICE,flush=True)
 for f in folds:
  for seed in cfg['seeds']:
   cm=camera_fit(cd,f,seed)
   for arm in cfg['arms']:person_fit(d,meta,f,seed,arm,cm)
 evaluate(d,meta,folds,cfg)
if __name__=='__main__':main()
