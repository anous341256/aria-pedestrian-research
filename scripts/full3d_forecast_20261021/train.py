import time,copy,hashlib,random
from geometry_common import *
from network import Camera,Person,torch,camera_velocity
import cv2
torch.set_num_threads(2);DEVICE='cuda' if torch.cuda.is_available() else 'cpu'
def tensor(x):return torch.as_tensor(x,dtype=torch.float32,device=DEVICE)
def seed_all(s):random.seed(s);np.random.seed(s);torch.manual_seed(s);torch.cuda.manual_seed_all(s)
def rot(x):return np.stack([cv2.Rodrigues(v)[0] for v in np.asarray(x,dtype=float).reshape(-1,3)]).reshape(np.asarray(x).shape[:-1]+(3,3))
def old_camera(fold,seed):return torch.load(BASE/'with_translation'/f'fold_{fold["fold"]}/seed_{seed}/camera/checkpoint.pt',map_location=DEVICE,weights_only=False)['model']
def camera_inputs(d,idx):
 key='rotation' if 'rotation' in d else 'cam'
 return np.concatenate([d[key][idx,:16],d['translation'][idx,:16]],-1).astype(np.float32),d['seconds'][idx,:16].astype(np.float32)
@torch.no_grad()
def camera_predict(model,d,idx):
 rr=[];bb=[]
 for ix in np.array_split(idx,max(1,int(np.ceil(len(idx)/1024)))):
  x,t=camera_inputs(d,ix);r,b=model(tensor(x),tensor(t));rr.append(r.cpu().numpy());bb.append(b.cpu().numpy())
 return np.concatenate(rr),np.concatenate(bb)
def features(d,idx):
 p=d['xyz'][idx,:16];valid=d['depth_valid'][idx,:16];assert valid.any(1).all();last=np.max(np.where(valid,np.arange(16),-1),axis=1);anchor=p[np.arange(len(idx)),last];rel=p-anchor[:,None];rel[~valid]=0;delta=np.zeros_like(rel);delta[:,1:]=(p[:,1:]-p[:,:-1])*(valid[:,1:]&valid[:,:-1])[...,None];tm=d['seconds'][idx,:16];age=-tm[np.arange(len(idx)),last]
 x=np.concatenate([rel,delta,valid[...,None],tm[...,None]/1.5,d['depth_quality'][idx,:16]],-1);ctx=np.concatenate([anchor,age[:,None]/1.5],-1)
 return x.astype(np.float32),ctx.astype(np.float32),anchor
def noise(meta,idx):return np.stack([np.random.default_rng(100000+int(hashlib.sha256((meta[i]['key']+'3Dfuture').encode()).hexdigest()[:8],16)).normal(0,.15,(5,15)) for i in idx]).astype(np.float32)
@torch.no_grad()
def person_predict(model,d,meta,idx):
 x,c,anchor=features(d,idx);return model.predict(tensor(x),tensor(c),tensor(noise(meta,idx))).cpu().numpy()+anchor[:,None]
def balanced(indices,meta,rng):
 groups=sorted({meta[i]['recording'] for i in indices});return np.array([rng.choice([i for i in indices if meta[i]['recording']==g]) for g in rng.choice(groups,48)])
def camera_fit(d,fold,seed):
 folder=OUT/f'fold_{fold["fold"]}/seed_{seed}/camera';folder.mkdir(parents=True,exist_ok=True);seed_all(seed);model=Camera(old_camera(fold,seed)).to(DEVICE)
 if (folder/'complete.json').exists():model.load_state_dict(torch.load(folder/'checkpoint.pt',map_location=DEVICE,weights_only=False)['model']);return model.eval()
 start=time.time();ti=np.array(fold['train_indices']);vi=np.array(fold['val_indices']);model.eval();r,b=camera_predict(model,d,ti);scales=np.maximum([np.mean((r-d['rotation'][ti,16:])**2),np.mean((b-d['translation'][ti,16:])**2)],1e-6)
 x,t=camera_inputs(d,ti);x=tensor(x);t=tensor(t);yr=tensor(d['rotation'][ti,16:]);yb=tensor(d['translation'][ti,16:]);opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
 def val():
  model.eval();r,b=camera_predict(model,d,vi);return float(np.mean((r-d['rotation'][vi,16:])**2)/scales[0]+np.mean((b-d['translation'][vi,16:])**2)/scales[1])
 best=val();state=copy.deepcopy(model.state_dict());beststep=0;logs=[dict(step=0,validation_joint_MSE=best)];rng=np.random.default_rng(seed);left=np.array([],int);draws=hashlib.sha256()
 for step in range(1,2001):
  while len(left)<256:left=np.concatenate([left,rng.permutation(len(ti))])
  ii,left=left[:256],left[256:];draws.update(ii.tobytes());model.train();r,b=model(x[ii],t[ii]);loss=(r-yr[ii]).square().mean()/scales[0]+(b-yb[ii]).square().mean()/scales[1];assert torch.isfinite(loss);opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
  if step%100==0:
   v=val();logs.append(dict(step=step,validation_joint_MSE=v))
   if v<best:best=v;beststep=step;state=copy.deepcopy(model.state_dict())
   write(OUT/'progress.json',dict(fold=fold['fold'],seed=seed,stage='camera',step=step))
 model.load_state_dict(state);model.eval();torch.save(dict(model=state,step=beststep,scales=scales.tolist(),fold=fold,seed=seed,protocol_sha256=sha(OUT/'protocol.json')),folder/'checkpoint.pt');write(folder/'training.json',logs);write(folder/'complete.json',dict(seconds=time.time()-start,step=beststep,scales=scales.tolist(),draws=draws.hexdigest(),parameters=sum(p.numel() for p in model.parameters())));print('CAMERA',fold['fold'],seed,beststep,round(time.time()-start,1),flush=True);return model
def person_fit(d,meta,fold,seed,cm):
 folder=OUT/f'fold_{fold["fold"]}/seed_{seed}/person';folder.mkdir(parents=True,exist_ok=True);seed_all(seed);model=Person().to(DEVICE)
 if (folder/'complete.json').exists():return
 start=time.time();vi=np.array(fold['val_indices']);vm=d['depth_valid'][vi,16:];assert vm.any();opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
 def val():
  model.eval();p=person_predict(model,d,meta,vi);return float(np.linalg.norm(p-d['xyz'][vi,16:],axis=-1)[vm].mean())
 best=val();state=copy.deepcopy(model.state_dict());beststep=0;logs=[dict(step=0,validation_3D_ADE_m=best)];draws=hashlib.sha256()
 for step in range(240):
  epoch,update=divmod(step,2);rng=np.random.default_rng(seed*100000+epoch*10+update);idx=balanced(fold['train_indices'],meta,rng);draws.update(idx.tobytes());x,c,anchor=features(d,idx);mask=np.repeat(d['depth_valid'][idx,:,None],3,-1);target=d['xyz'][idx]-anchor[:,None];target[~mask]=0
  t1=tensor(rng.random(48));t2=tensor(rng.random(48));n1=tensor(rng.normal(0,.15,(48,48)));n2=tensor(rng.normal(0,.15,(48,15)))
  model.train();loss=model.loss(tensor(x),tensor(c),tensor(target[:,:16].reshape(48,-1)),tensor(target[:,16:].reshape(48,-1)),tensor(mask[:,:16].reshape(48,-1)),tensor(mask[:,16:].reshape(48,-1)),t1,t2,n1,n2);assert torch.isfinite(loss);opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
  if (step+1)%20==0:
   v=val();logs.append(dict(step=step+1,validation_3D_ADE_m=v))
   if v<best:best=v;beststep=step+1;state=copy.deepcopy(model.state_dict())
   write(OUT/'progress.json',dict(fold=fold['fold'],seed=seed,stage='person',step=step+1))
 model.load_state_dict(state);model.eval();idx=np.array(fold['test_indices']);r,b=camera_predict(cm,d,idx);p=person_predict(model,d,meta,idx);z=diagnostics(p,r,b,d,idx);np.savez_compressed(folder/'predictions.npz',indices=idx,**z);torch.save(dict(model=state,fold=fold,seed=seed,step=beststep,protocol_sha256=sha(OUT/'protocol.json')),folder/'checkpoint.pt');write(folder/'training.json',logs);write(folder/'complete.json',dict(seconds=time.time()-start,step=beststep,draws=draws.hexdigest(),parameters=sum(p.numel() for p in model.parameters())));print('PERSON',fold['fold'],seed,beststep,round(time.time()-start,1),flush=True)
def diagnostics(p,r,b,d,idx):
 A=rot(r);truth=d['xy'][idx,16:];valid=d['depth_valid'][idx,16:];gtp=d['xyz'][idx,16:];gtA=d['A'][idx,16:];gtb=d['translation'][idx,16:];rel=np.swapaxes(A,-1,-2)@gtA;angles=np.rad2deg(np.linalg.norm(np.stack([cv2.Rodrigues(a)[0][:,0] for a in rel.reshape(-1,3,3)]).reshape(-1,5,3),axis=-1))
 z=dict(person_xyz=p,rotation=r,translation=b,valid_3D=valid,person_3D_error=np.linalg.norm(p-gtp,axis=-1),angle_deg=angles,translation_error=np.linalg.norm(b-gtb,axis=-1))
 for name,pp,aa,bb in [('image',p,A,b),('gt_camera',p,gtA,gtb),('gt_rotation',p,gtA,b),('gt_translation',p,A,gtb),('gt_person',gtp,A,b)]:
  xy,depth=project(to_camera(pp,aa,bb),d['K'][idx]);error=np.linalg.norm(xy-truth,axis=-1);bad=(depth<=1e-6)|~np.isfinite(error);error[bad]=32000.;z[name]=xy;z[name+'_depth']=depth;z[name+'_error']=error
 return z
def baselines(d,idx):
 x,c,anchor=features(d,idx);valid=d['depth_valid'][idx,:16];times=d['seconds'][idx,:16];p=d['xyz'][idx,:16];future=np.arange(1,6)*.1;hold=np.repeat(anchor[:,None],5,axis=1);cv=[]
 for j in range(len(idx)):
  t=times[j,valid[j]];v=p[j,valid[j]];center=t-t.mean();slope=(center[:,None]*v).sum(0)/max((center**2).sum(),1e-8);cv.append(anchor[j]+(future-t[-1])[:,None]*slope)
 return hold,np.array(cv)
def row(meta,i,seed,arm,z,j):
 r=dict(arm=arm,index=int(i),key=meta[i]['key'],recording=meta[i]['recording'],seed=seed,nonoverlap=meta[i]['nonoverlap'],ADE=float(z['image_error'][j].mean()),FDE=float(z['image_error'][j,-1]))
 if 'person_3D_error' in z:
  mask=z['valid_3D'][j];r.update(person_3D_ADE=float(z['person_3D_error'][j,mask].mean()) if mask.any() else None,person_3D_FDE=float(z['person_3D_error'][j,-1]) if mask[-1] else None,camera_mean_deg=float(z['angle_deg'][j].mean()),camera_final_deg=float(z['angle_deg'][j,-1]),camera_translation_ADE=float(z['translation_error'][j].mean()),camera_translation_FDE=float(z['translation_error'][j,-1]),gt_camera_ADE=float(z['gt_camera_error'][j].mean()),gt_rotation_ADE=float(z['gt_rotation_error'][j].mean()),gt_translation_ADE=float(z['gt_translation_error'][j].mean()),gt_person_ADE=float(z['gt_person_error'][j,mask].mean()) if mask.any() else None)
 return r
def summary(rows):
 keys=['ADE','FDE','person_3D_ADE','person_3D_FDE','camera_mean_deg','camera_final_deg','camera_translation_ADE','camera_translation_FDE','gt_camera_ADE','gt_rotation_ADE','gt_translation_ADE','gt_person_ADE'];keys=[k for k in keys if k in rows[0]]
 def avg(rr):return {k:float(np.mean([r[k] for r in rr if r.get(k) is not None])) if any(r.get(k) is not None for r in rr) else None for k in keys}
 rec={g:avg([r for r in rows if r['recording']==g]) for g in sorted({r['recording'] for r in rows})}
 return dict(window_mean=avg(rows),per_recording=rec,recording_equal={k:float(np.mean([v[k] for v in rec.values() if v[k] is not None])) for k in keys},per_seed={str(s):avg([r for r in rows if r['seed']==s]) for s in sorted({r['seed'] for r in rows})},sparse=avg([r for r in rows if r['nonoverlap']]))
def evaluate(d,meta,folds):
 rows=[];invalid=0;maxshared=0.;seconds=0.;steps={'camera':0,'person':0};replays=0
 short={k:(v[:,:16].copy() if v.ndim>=2 and v.shape[1]==21 else v.copy()) for k,v in d.items()};short['xyz']=to_fixed(lift(short['xy'],short['depth'],short['K']),short['A'],short['translation']);short['xyz'][~short['depth_valid']]=0
 for fold in folds:
  idx=np.array(fold['test_indices']);hold,cv=baselines(d,idx)
  for seed in [17,29,43]:
   folder=OUT/f'fold_{fold["fold"]}/seed_{seed}';cm=Camera(old_camera(fold,seed)).to(DEVICE);pm=Person().to(DEVICE);cm.load_state_dict(torch.load(folder/'camera/checkpoint.pt',map_location=DEVICE,weights_only=False)['model']);pm.load_state_dict(torch.load(folder/'person/checkpoint.pt',map_location=DEVICE,weights_only=False)['model']);cm.eval();pm.eval();r,b=camera_predict(cm,d,idx);p=person_predict(pm,d,meta,idx);rr,bb=camera_predict(cm,short,idx);pp=person_predict(pm,short,meta,idx);assert np.array_equal(r,rr) and np.array_equal(b,bb) and np.array_equal(p,pp);z=diagnostics(p,r,b,d,idx);stored=dict(np.load(folder/'person/predictions.npz'));assert all(np.array_equal(v,stored[k]) for k,v in z.items());replays+=2;invalid+=int((z['image_depth']<=1e-6).sum());rows += [row(meta,i,seed,'full3d',z,j) for j,i in enumerate(idx)]
   for name,xyz in [('fixed3D',hold),('CV3D',cv)]:
    zz=diagnostics(xyz,r,b,d,idx);rows += [row(meta,i,seed,name,zz,j) for j,i in enumerate(idx)]
   old=np.load(BASE/'with_translation'/f'fold_{fold["fold"]}/seed_{seed}/person/predictions.npz');assert np.array_equal(old['indices'],idx);rows += [row(meta,i,seed,'saved_translation',old,j) for j,i in enumerate(idx)]
   grouped={}
   for j,i in enumerate(idx):grouped.setdefault((meta[i]['recording'],meta[i]['timestamp_ns']),[]).append(j)
   for jj in grouped.values():maxshared=max(maxshared,float(np.max(abs(r[jj]-r[jj[0]]))),float(np.max(abs(b[jj]-b[jj[0]]))))
   for part in ['camera','person']:
    c=read(folder/part/'complete.json');seconds+=c['seconds'];steps[part]+=c['step']==0
 for i,m in enumerate(meta):
  e=np.linalg.norm(d['xy'][i,15]-d['xy'][i,16:],axis=-1);rows.append(row(meta,i,0,'hold_last',{'image_error':e[None]},0))
 results={a:summary([r for r in rows if r['arm']==a]) for a in ['full3d','fixed3D','CV3D','saved_translation','hold_last']};contrasts={}
 for ref in ['saved_translation','fixed3D','CV3D','hold_last']:
  a=np.array([np.mean([r['ADE'] for r in rows if r['arm']=='full3d' and r['index']==i]) for i in range(117)]);b=np.array([np.mean([r['ADE'] for r in rows if r['arm']==ref and r['index']==i]) for i in range(117)]);diff=a-b;rec=np.array([diff[[i for i,m in enumerate(meta) if m['recording']==g]].mean() for g in sorted({m['recording'] for m in meta})]);boot=np.random.default_rng(20261021).choice(rec,(5000,len(rec)),replace=True).mean(1);contrasts[ref]=dict(ADE_difference=float(diff.mean()),relative_percent=float((a.mean()/b.mean()-1)*100),improved_windows=int((diff<0).sum()),improved_recordings=int((rec<0).sum()),recording_bootstrap_95=np.quantile(boot,[.025,.975]).tolist())
 assert maxshared==0;protected=read(OUT/'source_hashes.json');assert all(sha(ROOT/k)==v for k,v in protected.items());write(OUT/'metrics.json',rows);write(OUT/'comparison.json',dict(results=results,contrasts=contrasts,training_seconds=seconds,step0_selected=steps));write(OUT/'verification.json',dict(passed=True,checkpoint_replays=replays,future_truncated_exact=True,history_xyz_rebuilt_from_truncated_sources=True,same_time_camera_max_difference=maxshared,invalid_rays=invalid,protected_files=len(protected)));print({k:v['window_mean'] for k,v in results.items()},flush=True)
def main():
 assert read(OUT/'geometry_preflight.json')['passed'] and read(OUT/'synthetic_geometry.json')['passed'];d=dict(np.load(OUT/'dataset.npz'));c=dict(np.load(OUT/'camera_dataset.npz'));meta=read(BASE/'samples.json');folds=read(BASE/'folds.json');cfolds=read(BASE/'camera_folds.json');write(OUT/'execution_code_hashes.json',{str(p.relative_to(ROOT)):sha(p) for p in Path(__file__).parent.glob('*.py')})
 for fold,cfold in zip(folds,cfolds):
  assert fold['test']==cfold['test'] and fold['validation']==cfold['validation']
  for seed in [17,29,43]:cm=camera_fit(c,cfold,seed);person_fit(d,meta,fold,seed,cm)
 evaluate(d,meta,folds);write(OUT/'training_complete.json',dict(passed=True,camera_fits=18,person_fits=18))
if __name__=='__main__':main()
