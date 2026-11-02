"""Frozen camera, matched person continuation on original versus quality-trim data."""
from forecast_common import *
import run as base
from network import Person,Camera,torch
torch.set_num_threads(2)

def load_base(fold,seed):
 source=BASE/'with_translation'/f'fold_{fold["fold"]}/seed_{seed}'
 cm=Camera().to(base.DEVICE);cm.load_state_dict(torch.load(source/'camera/checkpoint.pt',map_location=base.DEVICE,weights_only=False)['model']);cm.eval()
 for p in cm.parameters():p.requires_grad_(False)
 state=torch.load(source/'person/checkpoint.pt',map_location=base.DEVICE,weights_only=False)['model']
 return cm,state,source

def fit(old,meta,new,nmeta,fold,plan,seed,arm):
 folder=OUT/arm/f'fold_{fold["fold"]}/seed_{seed}';folder.mkdir(parents=True,exist_ok=True)
 if (folder/'complete.json').exists():return
 start=time.time();base.seed_all(seed);cm,state,source=load_base(fold,seed);ch=base.statehash(cm);pm=Person().to(base.DEVICE);pm.load_state_dict(state);pm.eval();initial=base.statehash(pm)
 data,tm,ti=(old,meta,fold['train_indices']) if arm==ARMS[0] else (new,nmeta,plan['trimmed_train_indices'])
 ti=np.array(ti);vi=np.array(fold['val_indices']);assert all(tm[i]['recording'] not in [fold['test'],fold['validation']] for i in ti)
 stable=warp(data['xy'],data['H'])[0]/base.SCALE;vtruth=warp(old['xy'][vi,16:21],old['H'][vi,16:21])[0]
 def val():
  pm.eval();z=base.predict(pm,cm,old,meta,vi,'with_translation');return float(np.linalg.norm(z['component']-vtruth,axis=-1).mean())
 best=val();stepbest=0;beststate=copy.deepcopy(pm.state_dict());logs=[dict(step=0,validation_residual_ADE=best)];opt=torch.optim.AdamW(pm.parameters(),lr=.0001,weight_decay=.01);draws=hashlib.sha256();grad=0.
 for step in range(240):
  epoch,update=divmod(step,2);rng=np.random.default_rng(seed*100000+epoch*10+update);idx=base.balanced(ti,tm,rng);draws.update(idx.tobytes());x,o=base.features(data,idx,'with_translation');y=stable[idx]-o[:,None]
  t1=base.tensor(rng.random(48));t2=base.tensor(rng.random(48));n1=base.tensor(rng.normal(0,.15,(48,32)));n2=base.tensor(rng.normal(0,.15,(48,10)))
  pm.train();loss,lf,lh=pm.loss(base.tensor(x),base.tensor(o),base.tensor(y[:,:16].reshape(48,-1)),base.tensor(y[:,16:21].reshape(48,-1)),t1,t2,n1,n2)
  assert torch.isfinite(loss);opt.zero_grad();loss.backward();grad=max(grad,float(pm.encoder.weight_ih_l0.grad.abs().max()));torch.nn.utils.clip_grad_norm_(pm.parameters(),1);opt.step()
  if (step+1)%20==0:
   v=val();logs.append(dict(step=step+1,validation_residual_ADE=v,training_loss=float(loss.detach().cpu())))
   if v<best:best=v;stepbest=step+1;beststate=copy.deepcopy(pm.state_dict())
   write(OUT/'training_progress.json',dict(arm=arm,fold=fold['fold'],seed=seed,step=step+1))
 pm.load_state_dict(beststate);pm.eval();idx=np.array(fold['test_indices']);z=base.diagnostics(base.predict(pm,cm,old,meta,idx,'with_translation'),old,idx);assert base.statehash(cm)==ch
 torch.save(dict(model=beststate,arm=arm,seed=seed,fold=fold,step=stepbest,protocol_sha256=sha(OUT/'protocol.json')),folder/'checkpoint.pt');np.savez_compressed(folder/'predictions.npz',indices=idx,**z)
 write(folder/'training.json',logs);write(folder/'complete.json',dict(seconds=time.time()-start,step=stepbest,initial_hash=initial,draws_hash=draws.hexdigest(),gradient_max=grad,camera_hash=ch,camera_source=str((source/'camera/checkpoint.pt').relative_to(ROOT)),training_indices=ti.tolist(),training_recordings=sorted({tm[i]['recording'] for i in ti}),parameters=sum(p.numel() for p in pm.parameters())))
 print(arm,fold['fold'],seed,'step',stepbest,'seconds',round(time.time()-start,1),flush=True)

def evaluate(data,meta,folds):
 rows=[];steps={a:0 for a in ARMS};seconds=0.;replays=0;invalid=0;shared={};initdiff=0.
 short={k:v[:,:16] if k in ['xy','raw','raw_valid','R','H','cam','seconds','boxes'] else v for k,v in data.items()}
 for fold in folds:
  idx=np.array(fold['test_indices'])
  for seed in [17,29,43]:
   cm,state,source=load_base(fold,seed);ref=dict(np.load(source/'person/predictions.npz'));assert np.array_equal(ref['indices'],idx)
   original=Person().to(base.DEVICE);original.load_state_dict(state);original.eval();originalpred=base.predict(original,cm,data,meta,idx,'with_translation');assert all(np.array_equal(originalpred[k],ref[k]) for k in originalpred)
   rows += [base.row(meta[i],i,seed,'saved_translation',ref,j) for j,i in enumerate(idx)];checks=[];vals=[]
   for arm in ARMS:
    folder=OUT/arm/f'fold_{fold["fold"]}/seed_{seed}';pm=Person().to(base.DEVICE);pm.load_state_dict(torch.load(folder/'checkpoint.pt',map_location=base.DEVICE,weights_only=False)['model']);pm.eval();stored=dict(np.load(folder/'predictions.npz'));z=base.predict(pm,cm,data,meta,idx,'with_translation');trunc=base.predict(pm,cm,short,meta,idx,'with_translation')
    assert np.array_equal(z['rotation'],ref['rotation'])
    for k in z:assert np.array_equal(z[k],stored[k]) and np.array_equal(z[k],trunc[k])
    z=base.diagnostics(z,data,idx);assert all(np.array_equal(z[k],stored[k]) for k in z);replays+=1;invalid+=int((z['image_depth']<=1e-6).sum())
    for j,i in enumerate(idx):
     key=(seed,meta[i]['recording'],meta[i]['timestamp_ns']);r=z['rotation'][j]
     if key in shared:assert np.array_equal(shared[key],r)
     else:shared[key]=r
    rows += [base.row(meta[i],i,seed,arm,z,j) for j,i in enumerate(idx)];c=read(folder/'complete.json');checks.append(c);seconds+=c['seconds'];steps[arm]+=c['step']==0;vals.append(read(folder/'training.json')[0]['validation_residual_ADE'])
   assert all(checks[0][k]==checks[1][k] for k in ['initial_hash','camera_hash','parameters']);assert vals[0]==vals[1];assert checks[0]['gradient_max']>0 and checks[1]['gradient_max']>0
 for i,m in enumerate(meta):
  e=np.linalg.norm(data['xy'][i,15]-data['xy'][i,16:21],axis=-1);rows.append(base.row(m,i,0,'hold_last',{'image_error':e[None]},0))
 results={a:base.summary([r for r in rows if r['arm']==a]) for a in ARMS+['saved_translation','hold_last']};contrasts={}
 for arm,ref in [(ARMS[1],ARMS[0]),(ARMS[1],'saved_translation'),(ARMS[0],'saved_translation')]:
  a=np.array([np.mean([r['ADE'] for r in rows if r['arm']==arm and r['index']==i]) for i in range(117)]);b=np.array([np.mean([r['ADE'] for r in rows if r['arm']==ref and r['index']==i]) for i in range(117)]);diff=a-b;rec=np.array([diff[[i for i,m in enumerate(meta) if m['recording']==g]].mean() for g in sorted({m['recording'] for m in meta})]);boot=np.random.default_rng(20261028).choice(rec,(5000,len(rec)),replace=True).mean(1)
  contrasts[arm+'_vs_'+ref]=dict(ADE_difference=float(diff.mean()),relative_percent=float((a.mean()/b.mean()-1)*100),improved_windows=int((diff<0).sum()),improved_recordings=int((rec<0).sum()),recording_bootstrap_95=np.quantile(boot,[.025,.975]).tolist())
 protected=read(OUT/'source_hashes.json');assert all(sha(ROOT/k)==v for k,v in protected.items())
 write(OUT/'metrics.json',rows);write(OUT/'comparison.json',dict(results=results,contrasts=contrasts,training_seconds=seconds,step0_selected=steps));write(OUT/'verification.json',dict(passed=True,checkpoint_replays=replays,original_checkpoint_replays=18,future_truncated_exact=True,paired_initializations=True,paired_initial_validation_exact=True,camera_exact_saved=True,same_time_camera_exact=True,invalid_rays=invalid,protected_files=len(protected)))
 print({a:r['window_mean'] for a,r in results.items()},flush=True)

def main():
 assert read(OUT/'preflight.json')['passed'];old=dict(np.load(BASE/'dataset.npz'));meta=read(BASE/'samples.json');new=dict(np.load(OUT/'trimmed_dataset.npz'));nm=read(OUT/'trimmed_samples.json');folds=read(OUT/'folds.json');plans=read(OUT/'fold_plan.json')
 write(OUT/'execution_code_hashes.json',{str(p.relative_to(ROOT)):sha(p) for p in Path(__file__).parent.glob('*.py')})
 for f,p in zip(folds,plans):
  assert f['fold']==p['fold']
  for seed in [17,29,43]:
   for arm in ARMS:fit(old,meta,new,nm,f,p,seed,arm)
 evaluate(old,meta,folds);write(OUT/'training_complete.json',dict(passed=True,person_fits=36,camera_fits=0))
if __name__=='__main__':main()
