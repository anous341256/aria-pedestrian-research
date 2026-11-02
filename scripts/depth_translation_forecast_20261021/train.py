"""Paired historical-depth fine tuning; fixed original translation camera."""
import sys,time,copy,hashlib
from depth_common import *
sys.path.insert(0,str(ROOT/'scripts/translation_forecast_20261020'))
import run as base
from network import Person,Camera,torch,nn
torch.set_num_threads(2)
ARMS=['no_depth','with_depth']
class DepthPerson(Person):
 def __init__(self,state):
  super().__init__();self.encoder=nn.GRU(15,64,batch_first=True)
  state=copy.deepcopy(state);w=state['encoder.weight_ih_l0'];state['encoder.weight_ih_l0']=torch.cat([w,torch.zeros((w.shape[0],6),device=w.device)],1)
  self.load_state_dict(state)
def inputs(data,depth,idx,arm):
 x,o=base.features(data,idx,'with_translation')
 d=depth[idx] if arm=='with_depth' else np.zeros_like(depth[idx])
 return np.concatenate([x,d],-1),o
@torch.no_grad()
def predict(pm,cm,data,depth,meta,idx,arm):
 x,o=inputs(data,depth,idx,arm)
 p=pm.predict(base.tensor(x),base.tensor(o),base.tensor(base.noise(meta,idx))).cpu().numpy()
 return dict(component=(p+o[:,None])*base.SCALE,rotation=base.infer_camera(cm,base.camera_inputs(data,idx,'with_translation')))
def load_base(fold,seed):
 folder=BASE/'with_translation'/f'fold_{fold["fold"]}/seed_{seed}'
 cm=Camera().to(base.DEVICE);cm.load_state_dict(torch.load(folder/'camera/checkpoint.pt',map_location=base.DEVICE,weights_only=False)['model']);cm.eval()
 for p in cm.parameters():p.requires_grad_(False)
 state=torch.load(folder/'person/checkpoint.pt',map_location=base.DEVICE,weights_only=False)['model']
 return cm,state,folder
def fit(data,depth,meta,fold,seed,arm):
 folder=OUT/arm/f'fold_{fold["fold"]}/seed_{seed}';folder.mkdir(parents=True,exist_ok=True)
 if (folder/'complete.json').exists():return
 start=time.time();base.seed_all(seed);cm,state,source=load_base(fold,seed);pm=DepthPerson(state).to(base.DEVICE);pm.eval();init=base.statehash(pm)
 original=Person().to(base.DEVICE);original.load_state_dict(state);original.eval();vi=np.asarray(fold['val_indices']);ti=np.asarray(fold['train_indices'])
 a=base.predict(original,cm,data,meta,vi,'with_translation');b=predict(pm,cm,data,depth,meta,vi,arm)
 initdiff=float(np.max(np.abs(a['component']-b['component'])));assert initdiff<.01,initdiff
 stable=base.warp(data['xy'],data['H'])[0]/base.SCALE
 def val():
  pm.eval();z=predict(pm,cm,data,depth,meta,vi,arm)
  return float(np.linalg.norm(z['component']-stable[vi,16:21]*base.SCALE,axis=-1).mean())
 best=val();beststep=0;beststate=copy.deepcopy(pm.state_dict());logs=[dict(step=0,validation_residual_ADE=best)]
 opt=torch.optim.AdamW(pm.parameters(),lr=.001,weight_decay=.01);draws=hashlib.sha256();noisehash=hashlib.sha256();grad=0
 for step in range(240):
  epoch,update=divmod(step,2);rng=np.random.default_rng(seed*100000+epoch*10+update);idx=base.balanced(ti,meta,rng);draws.update(idx.tobytes());x,o=inputs(data,depth,idx,arm);y=stable[idx]-o[:,None]
  t1=base.tensor(rng.random(48));t2=base.tensor(rng.random(48));n1=base.tensor(rng.normal(0,.15,(48,32)));n2=base.tensor(rng.normal(0,.15,(48,10)))
  for v in [x[:,:,:9],o,t1.cpu().numpy(),t2.cpu().numpy(),n1.cpu().numpy(),n2.cpu().numpy()]:noisehash.update(np.ascontiguousarray(v).tobytes())
  pm.train();loss,lf,lh=pm.loss(base.tensor(x),base.tensor(o),base.tensor(y[:,:16].reshape(48,-1)),base.tensor(y[:,16:21].reshape(48,-1)),t1,t2,n1,n2)
  assert torch.isfinite(loss);opt.zero_grad();loss.backward();grad=max(grad,float(pm.encoder.weight_ih_l0.grad[:,9:].abs().max()));torch.nn.utils.clip_grad_norm_(pm.parameters(),1);opt.step()
  if (step+1)%20==0:
   v=val();logs.append(dict(step=step+1,validation_residual_ADE=v))
   if v<best:best=v;beststep=step+1;beststate=copy.deepcopy(pm.state_dict())
   write(OUT/'training_progress.json',dict(arm=arm,fold=fold['fold'],seed=seed,step=step+1))
 pm.load_state_dict(beststate);pm.eval();idx=np.asarray(fold['test_indices']);z=base.diagnostics(predict(pm,cm,data,depth,meta,idx,arm),data,idx)
 torch.save(dict(model=beststate,arm=arm,seed=seed,fold=fold,step=beststep,protocol_sha256=sha(OUT/'protocol.json')),folder/'checkpoint.pt');np.savez_compressed(folder/'predictions.npz',indices=idx,**z)
 write(folder/'training.json',logs);write(folder/'complete.json',dict(seconds=time.time()-start,step=beststep,initial_hash=init,draws_hash=draws.hexdigest(),noise_hash=noisehash.hexdigest(),initial_component_max_difference_px=initdiff,depth_input_gradient_max=grad,parameters=sum(p.numel() for p in pm.parameters()),camera_source=str((source/'camera/checkpoint.pt').relative_to(ROOT))))
 print(arm,fold['fold'],seed,'step',beststep,'seconds',round(time.time()-start,1),flush=True)
def evaluate(data,depth,meta,folds):
 rows=[];maxinit=0.;steps={a:0 for a in ARMS};training=0.;replays=0;invalid=0
 for fold in folds:
  idx=np.asarray(fold['test_indices'])
  for seed in [17,29,43]:
   cm,state,source=load_base(fold,seed);checks=[];reference=dict(np.load(source/'person/predictions.npz'))
   assert np.array_equal(reference['indices'],idx)
   rows += [base.row(meta[i],i,seed,'saved_translation',reference,j) for j,i in enumerate(idx)]
   for arm in ARMS:
    folder=OUT/arm/f'fold_{fold["fold"]}/seed_{seed}';pm=DepthPerson(state).to(base.DEVICE);pm.load_state_dict(torch.load(folder/'checkpoint.pt',map_location=base.DEVICE,weights_only=False)['model']);pm.eval();stored=dict(np.load(folder/'predictions.npz'));z=predict(pm,cm,data,depth,meta,idx,arm)
    assert np.array_equal(z['rotation'],reference['rotation'])
    short={k:v[:,:16] if v.ndim>=2 and v.shape[1]==21 else v for k,v in data.items()};trunc=predict(pm,cm,short,depth,meta,idx,arm)
    for k in z:assert np.array_equal(z[k],stored[k]) and np.array_equal(z[k],trunc[k])
    if arm=='no_depth':
     changed=predict(pm,cm,data,depth+123,meta,idx,arm);assert all(np.array_equal(changed[k],z[k]) for k in z)
    z=base.diagnostics(z,data,idx);assert all(np.array_equal(z[k],stored[k]) for k in z);replays+=1;invalid+=int((z['image_depth']<=1e-6).sum())
    rows += [base.row(meta[i],i,seed,arm,z,j) for j,i in enumerate(idx)]
    c=read(folder/'complete.json');checks.append(c);training+=c['seconds'];steps[arm]+=c['step']==0;maxinit=max(maxinit,c['initial_component_max_difference_px'])
   assert all(checks[0][k]==checks[1][k] for k in ['initial_hash','draws_hash','noise_hash','parameters'])
   assert checks[0]['depth_input_gradient_max']==0
 for i,m in enumerate(meta):
  e=np.linalg.norm(data['xy'][i,15]-data['xy'][i,16:21],axis=-1);rows.append(base.row(m,i,0,'hold_last',{'image_error':e[None]},0))
 results={a:base.summary([r for r in rows if r['arm']==a]) for a in ARMS+['saved_translation','hold_last']};contrasts={}
 for ref in ['no_depth','saved_translation']:
  a=np.array([np.mean([r['ADE'] for r in rows if r['arm']=='with_depth' and r['index']==i]) for i in range(117)]);b=np.array([np.mean([r['ADE'] for r in rows if r['arm']==ref and r['index']==i]) for i in range(117)]);diff=a-b
  rec=np.array([diff[[i for i,m in enumerate(meta) if m['recording']==g]].mean() for g in sorted({m['recording'] for m in meta})]);boot=np.random.default_rng(20261021).choice(rec,(5000,len(rec)),replace=True).mean(1)
  contrasts[ref]=dict(ADE_difference=float(diff.mean()),relative_percent=float((a.mean()/b.mean()-1)*100),improved_windows=int((diff<0).sum()),improved_recordings=int((rec<0).sum()),recording_bootstrap_95=np.quantile(boot,[.025,.975]).tolist())
 protected=read(OUT/'source_hashes.json');assert all(sha(ROOT/k)==v for k,v in protected.items())
 write(OUT/'metrics.json',rows);write(OUT/'comparison.json',dict(results=results,contrasts=contrasts,training_seconds=training,step0_selected=steps))
 write(OUT/'verification.json',dict(passed=True,checkpoint_replays=replays,future_truncated_exact=True,no_depth_ignores_depth=True,paired_initializations_draws_noise=True,camera_exact_saved=True,initial_component_max_difference_px=maxinit,invalid_rays=invalid,protected_files=len(protected)))
 print({a:r['window_mean'] for a,r in results.items()},flush=True)
def main():
 assert read(OUT/'extraction_complete.json')['passed'];data=dict(np.load(BASE/'dataset.npz'));depth=np.load(OUT/'depth_features.npz')['features'];meta=read(BASE/'samples.json');folds=read(BASE/'folds.json')
 write(OUT/'execution_code_hashes.json',{str(p.relative_to(ROOT)):sha(p) for p in Path(__file__).parent.glob('*.py')})
 for fold in folds:
  for seed in [17,29,43]:
   for arm in ARMS:fit(data,depth,meta,fold,seed,arm)
 evaluate(data,depth,meta,folds)
 write(OUT/'training_complete.json',dict(passed=True,person_fits=36,camera_fits=0))
if __name__=='__main__':main()
